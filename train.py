import os
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from init import init
from utils import save_model, adjust_learning_rate


def forward_pass(x, y_reg, y_dontcare=None):
    x = x.to(device)
    y_reg = y_reg.to(device)

    # Encoder batch forward => feature pyramid
    losses = {}
    y_reg_ = net(x)

    # model predict at low resolution
    if args.y_step > 1:
        if args.ori_res_loss:
            # upsample model pred s.t. compute loss at full resolution
            ori_w = y_reg.shape[2]
            y_reg_ = F.interpolate(y_reg_, size=ori_w, mode='linear', align_corners=False)
            # dontcare: corners, two sides
            y_dontcare = y_dontcare.to(device)
            y_reg_ = y_reg_[~y_dontcare]
            y_reg = y_reg[~y_dontcare]
        else:
            # downsample gt s.t. compute loss at low resolution
            y_reg = (y_reg[:, :, args.y_step//2-1::args.y_step] + y_reg[:, :, args.y_step//2::args.y_step])/2

    total_pixel = np.prod(y_reg.shape)
    #  dontcare = (
        #  (y_reg_ < -1) & (y_reg < -1) |\
        #  (y_reg_ > 1) & (y_reg > 1)
    #  )

    if args.loss == 'l1':
        losses['total'] = (y_reg_ - y_reg).abs().sum() / total_pixel
        #  losses['total'] = (y_reg_ - y_reg)[~dontcare].abs().sum() / total_pixel
    elif args.loss == 'l2':
        losses['total'] = ((y_reg_ - y_reg)**2).sum() / total_pixel
    elif args.loss == 'berhu':
        l1 = (y_reg_ - y_reg).abs()
        T = args.huber_const
        l2 = ((y_reg_ - y_reg)**2 + T**2) / (2 * T)
        losses['total'] = torch.where(l1 <= T, l1, l2).sum() / total_pixel
    elif args.loss == 'huber':
        l1 = (y_reg_ - y_reg).abs()
        T = args.huber_const
        l2 = ((y_reg_ - y_reg)**2 + T**2) / (2 * T)
        losses['total'] = torch.where(l1 <= T, l2, l1).sum() / total_pixel
    else:
        raise NotImplementedError()

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
    iterator_train = iter(loader_train)
    for _ in trange(len(loader_train),
                    desc='Train ep%s' % ith_epoch, position=1):
        adjust_learning_rate(optimizer, args)
        args.cur_iter += 1
        if args.y_step > 1 and args.ori_res_loss:
            x, y_reg, y_cor, y_dontcare = next(iterator_train)
            losses = forward_pass(x, y_reg, y_dontcare)
        else:
            x, y_reg = next(iterator_train)
            losses = forward_pass(x, y_reg)
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
                    losses = forward_pass(x, y_reg, y_dontcare)
                else:
                    x, y_reg = next(iterator_valid)
                    losses = forward_pass(x, y_reg)
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
