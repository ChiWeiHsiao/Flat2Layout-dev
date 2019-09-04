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
    parser.add_argument('--y_step', type=int, default=1)
    parser.add_argument('--imgroot', default='datas/lsun/images')
    parser.add_argument('--gtpath', default='datas/lsun/validation.npz')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    #  for i in range(30, 100, 5):
        #  args.pth = os.path.join(os.path.dirname(args.pth), 'epoch_%d.pth' %i)
        #  print(args.pth)
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

        #  if gt['img_name'][ith] != 'sun_ahwzhlaeygrdxnzd':
            #  continue
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
            #  out_reg = net(x[None])
            output = net(x[None])
            if isinstance(output, tuple):
                out_reg, out_cor = output
                # plot ceil,floor corners at the bottom of image
                u_cor, d_cor = out_cor.cpu().numpy()[0].repeat(args.y_step, axis=-1)
                u_cor = u_cor.reshape([1,-1]).repeat(10, axis=0)
                d_cor = d_cor.reshape([1,-1]).repeat(10, axis=0)
                u_cor = u_cor.reshape([10, w, 1]).repeat(3, axis=2)
                d_cor = d_cor.reshape([10, w, 1]).repeat(3, axis=2)
                u_cor[..., [0,2]] = 0  # keep only G
                d_cor[..., [0,1]] = 0  # keep only B
                u_cor, d_cor = 255*u_cor, 255*d_cor
                rgb = np.vstack([rgb, u_cor, d_cor])
            else:
                out_reg = output
            np_reg = out_reg[0].cpu().numpy() / 2 + 0.5  # [-1, 1] => [0, 1]

        #  plt.imshow(rgb)
        plt.imshow(np.clip(rgb, 0, 1))
        if args.y_step > 1:
            x_coord = np.arange(start=args.y_step//2-0.5, stop=w, step=args.y_step)   
            plt.plot(x_coord, np_reg[0] * h, 'go--')
            plt.plot(x_coord, np_reg[1] * h, 'bo--')
        else:
            x_coord = np.arange(w)   
            plt.plot(x_coord, np_reg[0] * h, 'g-')
            plt.plot(x_coord, np_reg[1] * h, 'b-')
        plt.savefig(os.path.join(args.out, gt['img_name'][ith] + '.rgb.png'))
        #  plt.savefig(os.path.join(args.out, gt['img_name'][ith] + str(i) + '.rgb.png'))
        plt.clf()

