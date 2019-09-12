import os
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from init import init
from utils import save_model, adjust_learning_rate


def forward_pass(x, y_reg, y_cor, y_key, y_dontcare=None, mode='train'):
    x = x.to(device)
    y_reg = y_reg.to(device)
    y_cor = y_cor.to(device)
    y_key = y_key.to(device)

    # Encoder batch forward => feature pyramid
    losses = {}
    if args.pred_cor and args.pred_key:
        y_reg_, y_cor_, y_key_ = net(x)
    elif args.pred_cor:
        y_reg_, y_cor_ = net(x)
    else:
        y_reg_ = net(x)


    # Resize GT/pred if model predict at low resolution
    if args.y_step > 1:
        if not args.ori_res_cor:
            # downsample GT y_cor s.t. compute loss at low resolution
            B, C, W = y_cor.shape
            y_cor = y_cor.reshape([B, C, W//args.y_step, args.y_step] )
            y_cor = y_cor.sum(3).float()
            # downsample GT y_key s.t. compute loss at low resolution
            y_key = y_key.reshape([B, C, W//args.y_step, args.y_step] )
            y_key = y_key.sum(3).float()

        if args.ori_res_loss:
            # upsample Pred y_reg_ s.t. compute loss at full resolution
            ori_w = y_reg.shape[2]
            if args.upsample_lr_pad:
                pad_left_ = (2*y_reg_[...,[0]]-y_reg_[...,[1]])
                pad_right_ = (2*y_reg_[...,[-1]]-y_reg_[...,[-2]])
                y_reg_ = torch.cat([pad_left_, y_reg_, pad_right_], -1)
                y_reg_ = F.interpolate(y_reg_, scale_factor=args.y_step, mode='linear', align_corners=False)
                y_reg_ = y_reg_[..., args.y_step:-args.y_step].clamp(min=args.outy_val_up, max=args.outy_val_bt)
                # downsample GT to W/32, then upsample back to W
                if args.gt_down_upsample:
                    y_reg = (y_reg[:, :, args.y_step//2-1::args.y_step] + y_reg[:, :, args.y_step//2::args.y_step])/2
                    pad_left = (2*y_reg[...,[0]]-y_reg[...,[1]])
                    pad_right = (2*y_reg[...,[-1]]-y_reg[...,[-2]])
                    y_reg = torch.cat([pad_left, y_reg, pad_right], -1)
                    y_reg = F.interpolate(y_reg, scale_factor=args.y_step, mode='linear', align_corners=False)
                    y_reg = y_reg[..., args.y_step:-args.y_step].clamp(min=args.outy_val_up, max=args.outy_val_bt)
            else:
                y_reg_ = F.interpolate(y_reg_, size=ori_w, mode='linear', align_corners=False)
            # dontcare: corners, two sides
            if args.use_dontcare:
                y_dontcare = y_dontcare.to(device)
                y_reg_ = y_reg_[~y_dontcare]
                y_reg = y_reg[~y_dontcare]
        else:
            # downsample GT y_reg s.t. compute loss at low resolution
            y_reg = (y_reg[:, :, args.y_step//2-1::args.y_step] + y_reg[:, :, args.y_step//2::args.y_step])/2

    # Compute loss
    losses['total'] = 0
    if args.pred_cor:
        losses['y_cor'] = F.binary_cross_entropy_with_logits(y_cor_, y_cor, reduction='mean',
                pos_weight=torch.FloatTensor([args.pos_weight_cor]).to(device))
        if mode=='train' and args.septrain and (args.cur_iter <= 1/3*args.max_iters): losses['y_cor'] *= 0
        losses['total'] += args.weight_cor * losses['y_cor']

    if args.pred_key:
        losses['y_key'] = F.binary_cross_entropy_with_logits(y_key_, y_key, reduction='mean',
                pos_weight=torch.FloatTensor([args.pos_weight_cor]).to(device))
        if mode=='train' and args.septrain and (args.cur_iter <= 1/3*args.max_iters): losses['y_key'] *= 0
        losses['total'] += args.weight_key * losses['y_key']

    total_pixel = np.prod(y_reg.shape)
    if args.loss == 'l1':
        losses['y_reg'] = (y_reg_ - y_reg).abs().sum() / total_pixel
    elif args.loss == 'l2':
        losses['y_reg'] = ((y_reg_ - y_reg)**2).sum() / total_pixel
    elif args.loss == 'berhu':
        l1 = (y_reg_ - y_reg).abs()
        T = args.huber_const
        l2 = ((y_reg_ - y_reg)**2 + T**2) / (2 * T)
        losses['y_reg'] = torch.where(l1 <= T, l1, l2).sum() / total_pixel
    elif args.loss == 'huber':
        l1 = (y_reg_ - y_reg).abs()
        T = args.huber_const
        l2 = ((y_reg_ - y_reg)**2 + T**2) / (2 * T)
        losses['y_reg'] = torch.where(l1 <= T, l2, l1).sum() / total_pixel
    else:
        raise NotImplementedError()
    if mode=='train' and args.septrain and (args.cur_iter > 1/3*args.max_iters and args.cur_iter <= 2/3*args.max_iters): losses['y_reg'] *= 0
    if mode=='train' and args.no_reg_loss: losses['y_reg'] *= 0
    losses['total'] += losses['y_reg']

    if args.lap_order:
        inplane = (y_reg > -1) & (y_reg < 1)
        dx = 2 / (x.shape[3] - 1)
        my = y_reg[:, :, args.lap_order:-args.lap_order]
        ly = y_reg[:, :, :-args.lap_order*2]
        ry = y_reg[:, :, args.lap_order*2:]
        dy = (ly - my) + (ry - my)
        my_ = y_reg_[:, :, args.lap_order:-args.lap_order]
        ly_ = y_reg_[:, :, :-args.lap_order*2]
        ry_ = y_reg_[:, :, args.lap_order*2:]
        dy_ = (ly_ - my_) + (ry_ - my_)
        mask = inplane[:, :, args.lap_order:-args.lap_order] &\
               inplane[:, :, :-args.lap_order*2] &\
               inplane[:, :, args.lap_order*2:]
        losses['lap'] = (dy - dy_)[mask].abs().mean()
        losses['total'] = losses['total'] + 5 * losses['lap']

    # Other statistical metric
    with torch.no_grad():
        losses['l1'] = (y_reg_ - y_reg).abs().mean()

        valid = (y_reg >= -1) & (y_reg <= 1)
        valid_ = (y_reg_ >= -1) & (y_reg_ <= 1)
        tp = (valid & valid_).float().sum()
        fp = (~valid & valid_).float().sum()
        tn = (~valid & ~valid_).float().sum()
        fn = (valid & ~valid_).float().sum()
        if tn + fp > 0:
            losses['neg_r'] = tn / (tn + fp)
        if tn + fn > 0:
            losses['neg_p'] = tn / (tn + fn)

        if args.pred_cor:
            pos = (y_cor >= 0.5)
            pos_ = (y_cor_ >= 0)
            tp = (pos & pos_).float().sum() 
            fp = (~pos & pos_).float().sum() 
            tn = (~pos & ~pos_).float().sum()
            fn = (pos & ~pos_).float().sum()
            if tp + fn > 0:
                losses['cor_r'] = tp / (tp + fn)
            if tp + fp > 0:
                losses['cor_p'] = tp / (tp + fp)

        if args.pred_key:
            pos = (y_key >= 0.5)
            pos_ = (y_key_ >= 0)
            tp = (pos & pos_).float().sum() 
            fp = (~pos & pos_).float().sum() 
            tn = (~pos & ~pos_).float().sum()
            fn = (pos & ~pos_).float().sum()
            if tp + fn > 0:
                losses['key_r'] = tp / (tp + fn)
            if tp + fp > 0:
                losses['key_p'] = tp / (tp + fp)

    return losses

def gogo_train():
    net.train()
    if args.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(args.freeze_earlier_blocks + 1):
            for m in blocks[i]:
                m.eval()
    if args.freeze_bn:
        for m in net.feature_extractor.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
    if args.freeze_stage1:
        net.eval()
        net.corkey_1x1.train()
        net.corkey_reduce_height_module.train()
        net.corkey_bi_rnn.train()
        net.corkey_drop_out.train()
        net.corkey_linear.train()

    iterator_train = iter(loader_train)
    for _ in trange(len(loader_train),
                    desc='Train ep%s' % ith_epoch, position=1):
        adjust_learning_rate(optimizer, args)
        args.cur_iter += 1
        if args.y_step > 1 and args.ori_res_loss:
            x, y_reg, y_cor, y_key, y_dontcare = next(iterator_train)
            losses = forward_pass(x, y_reg, y_cor, y_key, y_dontcare, mode='train')
        else:
            x, y_reg, y_cor, y_key = next(iterator_train)
            losses = forward_pass(x, y_reg, y_cor, y_key, mode='train')
        for k, v in losses.items():
            k = 'train/%s' % k
            tb_writer.add_scalar(k, v.item(), args.cur_iter)
        loss = losses['total']

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def gogo_valid():
    net.eval()
    if loader_valid is not None:
        iterator_valid = iter(loader_valid)
        valid_loss = {}
        valid_num = 0
        for _ in trange(len(loader_valid),
                        desc='Valid ep%d' % ith_epoch, position=2):
            with torch.no_grad():
                if args.y_step > 1 and args.ori_res_loss:
                    x, y_reg, y_cor, y_key, y_dontcare = next(iterator_valid)
                    losses = forward_pass(x, y_reg, y_cor, y_key, y_dontcare, mode='valid')
                else:
                    x, y_reg, y_cor, y_key = next(iterator_valid)
                    losses = forward_pass(x, y_reg, y_cor, y_key, mode='valid')
            valid_num += x.size(0)
            for k, v in losses.items():
                valid_loss[k] = valid_loss.get(k, 0) + v.item() * x.size(0)

        print(dict([(k, v / valid_num) for k, v in valid_loss.items()]))
        for k, v in valid_loss.items():
            k = 'valid/%s' % k
            tb_writer.add_scalar(k, v / valid_num, ith_epoch)


if __name__ == '__main__':

    from args import parse_args
    print(' Experiment setting '.center(60, '='))
    args = parse_args()
    device, loader_train, loader_valid, net, model_kwargs, optimizer, tb_writer = init(args)

    if args.load_pretrain:
        state_dict = torch.load(args.load_pretrain)
        net.load_state_dict(state_dict['state_dict'], strict=False)
        ith_epoch = 0
        gogo_valid()

    print('=' * 60)

    # Start training
    for ith_epoch in trange(1, args.epochs + 1, desc='Epoch', unit='ep'):
        gogo_train()
        gogo_valid()

        # Periodically save model
        if ith_epoch % args.save_every == 0:
            save_model(
                net,
                os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % ith_epoch),
                args.__dict__,
                model_kwargs)
