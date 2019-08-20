import os
import numpy as np
from imageio import imread
from skimage.transform import resize
from tqdm import trange

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from init import init
from utils import load_trained_model
from dataset import normalize_rgb


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--imgroot', default='datas/lsun/images')
    parser.add_argument('--gtpath', default='datas/lsun/validation.npz')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda')

    net, kwargs = load_trained_model(args.pth)
    net = net.to(device)
    net.eval()
    print(kwargs)

    h = kwargs['main_h']
    w = kwargs['main_w']
    gt = np.load(args.gtpath)

    for ith in trange(len(gt['img_name'])):
        path = os.path.join(args.imgroot, gt['img_name'][ith])
        if os.path.isfile(path):
            pass
        elif os.path.isfile(path + '.jpg'):
            path = path + '.jpg'
        elif os.path.isfile(path + '.png'):
            path = path + '.png'
        else:
            raise Exception('%s found !??' % path)
        corner_c = np.array(gt['ceiling_corner'][ith]).reshape(-1, 2)
        corner_f = np.array(gt['floor_corner'][ith]).reshape(-1, 2)
        corner_cw = np.array(gt['ceiling_wall'][ith]).reshape(-1, 2)
        corner_fw = np.array(gt['floor_wall'][ith]).reshape(-1, 2)

        # Prepare input
        rgb = imread(path)[..., :3] / 255
        ori_h, ori_w = rgb.shape[:2]
        if (ori_h, ori_w) != (h, w):
            rgb = resize(rgb, [h, w], preserve_range=True, anti_aliasing=False, mode='reflect')
        x = torch.FloatTensor(normalize_rgb(rgb).transpose(2, 0, 1).copy()).to(device)

        # Prepare output
        with torch.no_grad():
            out_reg, out_att = net(x[None])
            np_reg = out_reg[0].cpu().numpy() / 2 + 0.5  # [-1, 1] => [0, 1]
            np_att = out_att[0].cpu().numpy()

        att_rgb = np.zeros((*np_att.shape[1:], 3), np.float32)
        att_rgb[..., 1] = np_att[0]
        att_rgb[..., 2] = np_att[1]
        plt.imshow(rgb)
        plt.plot(np.arange(w), np_reg[0] * h, 'g-')
        plt.plot(np.arange(w), np_reg[1] * h, 'b-')
        plt.savefig(os.path.join(args.out, gt['img_name'][ith] + '.rgb.png'))
        plt.clf()

        plt.imshow(att_rgb)
        # plt.plot(np.arange(w), np_reg[0] * att_rgb.shape[0], 'g-')
        # plt.plot(np.arange(w), np_reg[1] * att_rgb.shape[0], 'b-')
        plt.savefig(os.path.join(args.out, gt['img_name'][ith] + '.att.png'))
        plt.clf()
