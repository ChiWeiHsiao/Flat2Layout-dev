import os
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from init import init
from utils import save_model


def forward_pass(x, y_reg):
    x = x.to(device)
    y_reg = y_reg.to(device)

    # Encoder batch forward => feature pyramid
    losses = {}
    y_reg_, y_att_, y_gain_ = net(x)
    total_pixel = np.prod(y_reg.shape)
    dontcare = (
        (y_reg_ < -1) & (y_reg < -1) |\
        (y_reg_ > 1) & (y_reg > 1)
    )

    if args.loss == 'l1':
        losses['total'] = (y_reg_ - y_reg)[~dontcare].abs().sum() / total_pixel
    elif args.loss == 'l2':
        losses['total'] = ((y_reg_ - y_reg)[~dontcare]**2).sum() / total_pixel
    elif args.loss == 'berhu':
        l1 = (y_reg_ - y_reg).abs()
        T = l1.max().item() * 0.2
        l2 = ((y_reg_ - y_reg)**2 + T**2) / (2 * T)
        losses['total'] = torch.where(l1 <= T, l1, l2).sum() / total_pixel
    elif args.loss == 'huber':
        l1 = (y_reg_ - y_reg).abs()
        T = l1.max().item() * 0.2
        l2 = ((y_reg_ - y_reg)**2 + T**2) / (2 * T)
        losses['total'] = torch.where(l1 <= T, l2, l1).sum() / total_pixel
    else:
        raise NotImplementedError()

    if args.guide_gain:
        y_gain = (y_reg > -1) & (y_reg < 1)
        y_gain_ = F.interpolate(y_gain_, y_gain.shape[2], mode='linear', align_corners=True)
        y_gain_ = torch.clamp(y_gain_, 1e-4, 1-1e-4)
        losses['gain_bce'] = F.binary_cross_entropy(y_gain_, y_gain.float())
        losses['total'] = losses['total'] + losses['gain_bce']

    # Other statistical metric
    with torch.no_grad():
        y_gain_ = F.interpolate(y_gain_, y_reg.shape[2], mode='linear', align_corners=True)
        pred_pos = (y_gain_ > 0.5)
        gt_pos = (y_reg > -1) & (y_reg < 1)
        tp = (pred_pos & gt_pos).float().sum()
        tn = (~pred_pos & ~gt_pos).float().sum()
        fp = (pred_pos & ~gt_pos).float().sum()
        fn = (~pred_pos & gt_pos).float().sum()
        losses['gain_recall'] = tp / (tp + fn)
        losses['gain_i_recall'] = tn / (tn + fp)

    return losses

def gogo_train():
    net.train()
    if args.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = net.encoder.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(args.freeze_earlier_blocks + 1):
            for m in blocks[i]:
                m.eval()
    if args.freeze_bn:
        for m in net.encoder.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
    iterator_train = iter(loader_train)
    for _ in trange(len(loader_train),
                    desc='Train ep%s' % ith_epoch, position=1):
        args.cur_iter += 1
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
            x, y_reg = next(iterator_valid)
            with torch.no_grad():
                losses = forward_pass(x, y_reg)
            valid_num += x.size(0)
            for k, v in losses.items():
                valid_loss[k] = valid_loss.get(k, 0) + v.item() * x.size(0)

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
    args.cur_iter = 0
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
