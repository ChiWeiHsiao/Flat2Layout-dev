import numpy as np
from math import sqrt
from collections import OrderedDict

import torch
import torch.nn.functional as F

import model


def save_model(net, path, kwargs, model_kwargs):
    torch.save(OrderedDict({
        'state_dict': net.state_dict(),
        'kwargs': kwargs,
        'model_kwargs': model_kwargs,
    }), path)


def load_trained_model(path):
    state_dict = torch.load(path, map_location='cpu')
    kwargs = state_dict['kwargs']
    Net = getattr(model, kwargs['net'])
    net = Net(**state_dict['model_kwargs'])
    try:
        net.load_state_dict(state_dict['state_dict'])
    except:
        import sys
        print(sys.exc_info())
        net.load_state_dict(state_dict['state_dict'], strict=False)
    return net, kwargs


if __name__ == '__main__':
    pass
