import os
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from init import init
from utils import save_model, adjust_learning_rate


def forward_pass(x, y_reg, y_cor, y_dontcare=None):
    x = x.to(device)
    y_reg = y_reg.to(device)
    y_cor = y_cor.to(device)

    # Encoder batch forward => feature pyramid
    losses = {}
    y_reg_, y_cor_ = net(x)

    # Resize GT/pred if model predict at low resolution
    if args.y_step > 1:
        # downsample GT y_cor s.t. compute loss at low resolution
        B, C, W = y_cor.shape
        y_cor = y_cor.reshape([B, C, W//args.y_step, args.y_step] )
        y_cor = y_cor.sum(3).float()
        if args.ori_res_loss:
            # upsample Pred y_reg_ s.t. compute loss at full resolution
            ori_w = y_reg.shape[2]
            y_reg_ = F.interpolate(y_reg_, size=ori_w, mode='linear', align_corners=False)
            # dontcare: corners, two sides
            y_dontcare = y_dontcare.to(device)
            y_reg_ = y_reg_[~y_dontcare]
            y_reg = y_reg[~y_dontcare]
        else:
            # downsample GT y_reg s.t. compute loss at low resolution
            y_reg = (y_reg[:, :, args.y_step//2-1::args.y_step] + y_reg[:, :, args.y_step//2::args.y_step])/2

    # Compute loss
    losses['total'] = 0
    losses['y_cor'] = F.binary_cross_entropy_with_logits(y_cor_, y_cor, reduction='mean',
                pos_weight=torch.FloatTensor([args.pos_weight_cor]).to(device))
    losses['total'] += args.weight_cor * losses['y_cor']

    total_pixel = np.prod(y_reg.shape)

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

    return losses

def gogo_train():
    net.train()
    for m in net.feature_extractor.modules():
        m.eval()
    for m in net.reduce_height_module.modules():
        m.eval()
    for m in net.bi_rnn.modules():
        m.eval()
    for m in net.linear.modules():
        m.eval()
    iterator_train = iter(loader_train)
    for _ in trange(len(loader_train),
                    desc='Train ep%s' % ith_epoch, position=1):
        adjust_learning_rate(optimizer, args)
        args.cur_iter += 1
        if args.y_step > 1 and args.ori_res_loss:
            x, y_reg, y_cor, y_dontcare = next(iterator_train)
            losses = forward_pass(x, y_reg, y_cor, y_dontcare)
        else:
            x, y_reg, y_cor = next(iterator_train)
            losses = forward_pass(x, y_reg, y_cor)
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
                    x, y_reg, y_cor, y_dontcare = next(iterator_valid)
                    losses = forward_pass(x, y_reg, y_cor, y_dontcare)
                else:
                    x, y_reg, y_cor = next(iterator_valid)
                    losses = forward_pass(x, y_reg, y_cor)
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
    print('=' * 60)

    state_dict = torch.load('ckpt/lowr_oriloss_horizon_y1y2/epoch_100.pth', map_location='cpu')
    net.load_state_dict(state_dict['state_dict'], strict=False)
    del state_dict

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
