import os
import numpy as np
from imageio import imread
from skimage.transform import resize

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from utils import find_invalid_data


# TODO: random crop


def normalize_rgb(rgb):
    rgb_mean = np.array([[[0.485, 0.456, 0.406]]])
    rgb_std = np.array([[[0.229, 0.224, 0.225]]])
    return (rgb - rgb_mean) / rgb_std

def undo_normalize_rgb(rgb):
    rgb_mean = np.array([[[0.485, 0.456, 0.406]]])
    rgb_std = np.array([[[0.229, 0.224, 0.225]]])
    return rgb * rgb_std + rgb_mean

def gen_1d_corner(xys, w):
    # xys in [0, 1]
    cor_1d = np.zeros(w, np.float32)
    if len(xys) <= 2:
        return cor_1d
    x = xys[1:-1, 0] * w
    cor_1d[x.astype(int)] = 1
    return cor_1d

def gen_1d(xys, w, missing_val, mode='constant'):
    '''  generate 1d boundary GT
    Input:
        xys: xy coordinates of keypoints

    Mode: setting value of y when it is outside image plane
        constant: set to a constant, missing_val
        linear: linearly grow to missing_val
    '''
    reg_1d = np.zeros(w, np.float32) + missing_val
    for i in range(1, len(xys)):
        x0, y0 = xys[i-1]
        x1, y1 = xys[i]
        x0 = x0 * w           # [0, 1] => [0, w]
        x1 = x1 * w           # [0, 1] => [0, w]
        y0 = (y0 - 0.5) * 2   # [0, 1] => [-1, 1]
        y1 = (y1 - 0.5) * 2   # [0, 1] => [-1, 1]
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        s = int(max(x0-1, 0))
        e = int(min(x1+1, w))
        reg_1d[s:e] = np.interp(np.arange(s, e), [x0, x1], [y0, y1])

    if len(xys)>0 and mode == 'linear':
        xst = 0
        x0, y0 = xys[0]
        if x0 >= 0:
            x1, y1 = xys[1]
            x0 = x0 * w           # [0, 1] => [0, w]
            x1 = x1 * w           # [0, 1] => [0, w]
            y0 = (y0 - 0.5) * 2   # [0, 1] => [-1, 1]
            y1 = (y1 - 0.5) * 2   # [0, 1] => [-1, 1]
            if x0-x1==0:
                raise ValueError('x0-x1==0')
            yst = y0+(y0-y1)/(x0-x1)*(xst-x0)
            #  if len(np.arange(xst, int(x0))) != len(reg_1d[xst:int(x0)]):
                #  print('corners:', xys)
                #  print('xst, yst:', xst, yst)
                #  print('x0, y0:', x0, y0)
            reg_1d[xst:int(x0)] = np.interp(np.arange(xst, int(x0)), [xst, x0], [yst, y0])

        xend = int(w)
        x0, y0 = xys[-2]
        x1, y1 = xys[-1]
        if x1 <= w:
            x0 = x0 * w           # [0, 1] => [0, w]
            x1 = x1 * w           # [0, 1] => [0, w]
            y0 = (y0 - 0.5) * 2   # [0, 1] => [-1, 1]
            y1 = (y1 - 0.5) * 2   # [0, 1] => [-1, 1]
            yend = y0+(y0-y1)/(x0-x1)*(xend-x0)
            reg_1d[int(min(x1+1, w)):xend] = np.interp(np.arange(int(min(x1+1, w)), xend), [x1, xend], [y1, yend])
        if missing_val > 1:
            reg_1d = np.clip(reg_1d, None, missing_val)
        else:
            reg_1d = np.clip(reg_1d, missing_val, None)
    return reg_1d

class FlatLayoutDataset(Dataset):
    def __init__(self, imgroot, gtpath, hw=(512, 512),
                 flip=False, gamma=False, outy_mode='constant', outy_val=(-1.05,1.05),
                 y_step=1, gen_doncare=False):
        gt = np.load(gtpath)
        self.gt_path = []
        for name in gt['img_name']:
            path = os.path.join(imgroot, name)
            if os.path.isfile(path):
                self.gt_path.append(path)
            elif os.path.isfile(path + '.jpg'):
                self.gt_path.append(path + '.jpg')
            elif os.path.isfile(path + '.png'):
                self.gt_path.append(path + '.png')
            else:
                raise Exception('%s found !??' % path)
        self.corner_c = [np.array(c).reshape(-1, 2) for c in gt['ceiling_corner']]
        self.corner_f = [np.array(c).reshape(-1, 2) for c in gt['floor_corner']]
        self.corner_cw = [np.array(c).reshape(-1, 2) for c in gt['ceiling_wall']]
        self.corner_fw = [np.array(c).reshape(-1, 2) for c in gt['floor_wall']]
        self.hw = hw
        self.flip = flip
        self.gamma = gamma
        self.outy_mode = outy_mode
        self.outy_val = outy_val
        self.y_step = y_step
        self.gen_doncare = gen_doncare

    def __len__(self):
        return len(self.gt_path)

    def __getitem__(self, idx):
        rgb = imread(self.gt_path[idx])[..., :3] / 255
        cc = self.corner_c[idx].astype(np.float32)    # Copy a new one
        cf = self.corner_f[idx].astype(np.float32)    # Copy a new one
        ccw = self.corner_cw[idx].astype(np.float32)  # Copy a new one
        cfw = self.corner_fw[idx].astype(np.float32)  # Copy a new one

        # Data augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 1:
                p = 1 / p
            rgb = rgb ** p
        if self.flip and np.random.randint(2) == 1:
            rgb, cc, cf, ccw, cfw = self._flip(rgb, cc, cf, ccw, cfw)

        # Finally normalize rgb and corners
        rgb, cc, cf, ccw, cfw = self._final_normalize(rgb, cc, cf, ccw, cfw)

        # Generate 1d regression gt
        u_1d = gen_1d(cc, rgb.shape[1], missing_val=self.outy_val[0], mode=self.outy_mode)
        d_1d = gen_1d(cf, rgb.shape[1], missing_val=self.outy_val[1], mode=self.outy_mode)
        u_1d_corner = gen_1d_corner(cc, rgb.shape[1])
        d_1d_corner = gen_1d_corner(cf, rgb.shape[1])

        # To tensor
        x = torch.FloatTensor(rgb.transpose(2, 0, 1).copy())
        y_reg = torch.FloatTensor([u_1d, d_1d])
        y_cor = torch.FloatTensor([u_1d_corner, d_1d_corner])

        if self.y_step > 1 and self.gen_doncare:
            y_dontcare = self._gen_doncare_mask(cc, cf, self.y_step)
            return x, y_reg, y_cor, y_dontcare
        else:
            return x, y_reg, y_cor

    def _gen_doncare_mask(self, cc, cf, y_step):
        # shape [2, W]
        S = y_step
        W = self.hw[1]
        # Corners: only use real corners, remove left right keypoints
        c_xs = cc[1:-1, 0] * self.hw[1]
        f_xs = cf[1:-1, 0] * self.hw[1]

        start = (c_xs-(S//2-0.5))//S * S + (S//2-0.5)
        stop = (c_xs-(S//2-0.5))//S * S + (S//2-0.5) + S
        start = np.clip(np.ceil(start).astype(int), a_min=0, a_max=W)
        stop = np.clip(np.ceil(stop).astype(int), a_min=0, a_max=W)
        c_mask = np.zeros(W, dtype=np.bool)
        for i in range(len(c_xs)):
            c_mask[start[i] : stop[i]] = True

        start = (f_xs-(S//2-0.5))//S * S + (S//2-0.5)
        stop = (f_xs-(S//2-0.5))//S * S + (S//2-0.5) + S
        start = np.clip(np.ceil(start).astype(int), a_min=0, a_max=W)
        stop = np.clip(np.ceil(stop).astype(int), a_min=0, a_max=W)
        f_mask = np.zeros(W , dtype=np.bool)
        for i in range(len(f_xs)):
            f_mask[start[i] : stop[i]] = True

        mask = np.vstack([c_mask, f_mask])

        # Leftmost and rightmost
        mask[:, :S//2] = True
        mask[:, -S//2:] = True
        return mask


    def _final_normalize(self, rgb, *corners_lst):
        # Rescale image to fix size
        # Rescale xy of all corners to [0, 1]
        ori_hw = rgb.shape[:2]
        rgb = normalize_rgb(resize(rgb, self.hw, preserve_range=True, anti_aliasing=False, mode='reflect'))
        for cs in corners_lst:
            cs[:, 0] /= ori_hw[1]  # Rescale x to [0, 1]
            cs[:, 1] /= ori_hw[0]  # Rescale y to [0, 1]
        return [rgb, *corners_lst]

    def _flip(self, rgb, *corners_lst):
        rgb = np.flip(rgb, 1)
        for cs in corners_lst:
            cs[:, 0] = rgb.shape[1] - cs[:, 0]  # Flip x
            cs[:] = np.flip(cs, 0)
        return [rgb, *corners_lst]


if __name__ == '__main__':

    import argparse
    from tqdm import trange
    from torch.utils.data import DataLoader
    import matplotlib; matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import convolve1d

    parser = argparse.ArgumentParser()
    parser.add_argument('--imgroot', default='datas/lsun/images_640x640')
    parser.add_argument('--gtpath', default='datas/lsun/training_640x640.npz')
    args = parser.parse_args()
    os.makedirs('vis/dataset', exist_ok=True)

    YSTEP = 32
    OUTY_MODE = 'constant'
    #  OUTY_MODE = 'linear'
    dataset = FlatLayoutDataset(args.imgroot, args.gtpath, hw=(640, 640), flip=True, outy_mode=OUTY_MODE, outy_val=(-1.1,1.1),
                                y_step=YSTEP, gen_doncare=True)

    #  ed, ey = find_invalid_data(args.gtpath)
    for i in trange(len(dataset)):
        #  error_lst = ed + ey
        #  if dataset.gt_path[i].split('/')[-1][:-4] not in error_lst:
            #  continue

        #  x, y_reg = dataset[i]
        x, y_reg, y_cor, y_dontcare = dataset[i]

        rgb = np.clip(undo_normalize_rgb(x.permute(1, 2, 0).numpy()), 0, 1)
        u_1d, d_1d = y_reg.numpy()
        u_1d_corner, d_1d_corner = y_cor.numpy()

        u_1d_xs = np.where(u_1d_corner)[0]
        d_1d_xs = np.where(d_1d_corner)[0]

        plt.imshow(rgb)
       # plot y_reg
        plt.plot(np.arange(rgb.shape[1]), (u_1d / 2 + 0.5) * rgb.shape[0], 'b-')
        plt.plot(np.arange(rgb.shape[1]), (d_1d / 2 + 0.5) * rgb.shape[0], 'g-')

        # plot low resolution reg
        if YSTEP > 1:
            lowr_y_reg = (y_reg[:, YSTEP//2-1::YSTEP] + y_reg[:, YSTEP//2::YSTEP])/2
            lowr_u_1d, lowr_d_1d = lowr_y_reg.numpy()
            x_cor = np.arange(rgb.shape[1])
            lowr_x_cor = (x_cor[YSTEP//2-1::YSTEP] + x_cor[YSTEP//2::YSTEP])/2
            plt.plot(lowr_x_cor, (lowr_u_1d / 2 + 0.5) * rgb.shape[0], 'bo')
            plt.plot(lowr_x_cor, (lowr_d_1d / 2 + 0.5) * rgb.shape[0], 'go')
        # plot high resolution reg upsapled from low resolution reg 
        if YSTEP > 1:
            ori_w = y_reg.shape[1]
            upsample_y_reg = F.interpolate(lowr_y_reg.reshape([1,2,-1]), size=ori_w, mode='linear', align_corners=False)
            upsample_y_reg = upsample_y_reg.reshape([2, -1])
            upsample_u_1d, upsample_d_1d = upsample_y_reg.numpy()
            plt.plot(np.arange(rgb.shape[1]), (upsample_u_1d / 2 + 0.5) * rgb.shape[0], 'r-')
            plt.plot(np.arange(rgb.shape[1]), (upsample_d_1d / 2 + 0.5) * rgb.shape[0], 'r-')
        # plot doncare
        x_cor = np.arange(rgb.shape[1])
        u_1d, d_1d = y_reg.numpy()
        plt.plot(x_cor[y_dontcare[0]], (y_reg[0, y_dontcare[0]] / 2 + 0.5) * rgb.shape[0], 'yo')
        plt.plot(x_cor[y_dontcare[1]], (y_reg[1, y_dontcare[1]] / 2 + 0.5) * rgb.shape[0], 'yo')

        plt.plot(u_1d_xs, np.zeros_like(u_1d_xs)+rgb.shape[0]//2-5, 'bo')
        plt.plot(d_1d_xs, np.zeros_like(d_1d_xs)+rgb.shape[0]//2+5, 'go')
        plt.savefig('vis/dataset/%s.vis.png' % (dataset.gt_path[i].split('/')[-1][:-4]))
        plt.clf()
