import numpy as np
from math import sqrt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import model
import model_HorizonNet


def find_invalid_data(gt_path):
    gt = np.load(gt_path)
    f = gt['floor_corner']
    c = gt['ceiling_corner']
    name = gt['img_name']
    N = f.shape[0]
    lst_dup = []
    lst_err_y = []
    for i in range(N):
        cxs = c[i][:, 0]
        cys = c[i][:, 1]
        fxs = f[i][:, 0]
        fys = f[i][:, 1]
        if len(cxs) != len(np.unique(cxs)) or len(fxs) != len(np.unique(fxs)):
            lst_dup.append(name[i])
        if np.any(cys >= 640-1) or np.any(fys <= 0):
            lst_err_y.append(name[i])
    return lst_dup, lst_err_y



def save_model(net, path, kwargs, model_kwargs):
    torch.save(OrderedDict({
        'state_dict': net.state_dict(),
        'kwargs': kwargs,
        'model_kwargs': model_kwargs,
    }), path)


def load_trained_model(path):
    state_dict = torch.load(path, map_location='cpu')
    kwargs = state_dict['kwargs']
    if kwargs['net'] in ['HorizonNet', 'LowResHorizonNet', 'TwoStageNet', 'LowHighNet']:
        Net = getattr(model_HorizonNet, kwargs['net'])
    else:
        Net = getattr(model, kwargs['net'])
    net = Net(**state_dict['model_kwargs'])
    try:
        net.load_state_dict(state_dict['state_dict'])
    except:
        import sys
        print(sys.exc_info())
        net.load_state_dict(state_dict['state_dict'], strict=False)
    return net, kwargs


def params_to_1d(params, W=1024):
    '''
    params: B x 4 x W'
        - params[:, 0] is v1 in [-pi/2, pi/2]
        - params[:, 1] is v2 in [-pi/2, pi/2]
        - params[:, 2] is v4 in [-pi/2, pi/2]
        - params[:, 3] is u4 in [-1, 1]
    return: B x 1 x W
    '''
    B, _, W_ = params.shape
    assert _ == 4 and W % W_ == 0

    # Assign zero shifted (u3 = 0) u1v1u2v2 for each output element
    v1 = torch.zeros([B, 1, W])
    v2 = torch.zeros([B, 1, W])
    u1 = torch.zeros([B, 1, W])
    u2 = torch.zeros([B, 1, W])

    # Generate 1d
    ones = torch.zeros([B, 1, W, 3])
    ones[..., 1] = 1
    xyz1 = torch.stack([
        torch.cos(v1) * torch.cos(u1),
        torch.cos(v1) * torch.sin(u1),
        torch.sin(v1),
    ])  # B x 1 x W x 3
    xyz2 = torch.stack([
        torch.cos(v2) * torch.cos(u2),
        torch.cos(v2) * torch.sin(u2),
        torch.sin(v2),
    ])  # B x 1 x W x 3
    cross = torch.cross(xyz1, xyz2, dim=3)
    cross = torch.cross(cross, ones, dim=3)
    assert cross[..., 1].abs().max().item() < 1e-6


def adjust_learning_rate(optimizer, args):
    if args.cur_iter < args.warmup_iters:
        frac = args.cur_iter / args.warmup_iters
        step = args.lr - args.warmup_lr
        args.running_lr = args.warmup_lr + step * frac
    else:
        frac = (float(args.cur_iter) - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        scale_running_lr = max((1. - frac), 0.) ** args.lr_pow
        args.running_lr = args.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr


def group_weight(module):
    # Group module parameters into two group
    # One need weight_decay and the other doesn't
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.RNNBase):
            for name, ps in m.named_parameters():
                if 'bias' in name:
                    group_no_decay.append(ps)
                else:
                    group_decay.append(ps)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    
    # Remove weights dont requires grad
    group_decay = [p for p in group_decay if p.requires_grad]
    group_no_decay = [p for p in group_no_decay if p.requires_grad]
    
    return [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]


if __name__ == '__main__':
    pass
