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
                 flip=False, gamma=False, outy_mode='constant'):
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
        u_1d = gen_1d(cc, rgb.shape[1], missing_val=-1.1, mode=self.outy_mode)
        d_1d = gen_1d(cf, rgb.shape[1], missing_val=1.1, mode=self.outy_mode)

        # To tensor
        x = torch.FloatTensor(rgb.transpose(2, 0, 1).copy())
        y_reg = torch.FloatTensor([u_1d, d_1d])
        return x, y_reg

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
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--imgroot', default='datas/lsun/images_640x640')
    parser.add_argument('--gtpath', default='datas/lsun/training_640x640.npz')
    args = parser.parse_args()
    os.makedirs('vis/dataset', exist_ok=True)

    dataset = FlatLayoutDataset(args.imgroot, args.gtpath, flip=True, outy_mode='linear')

    #  ed, ey = find_invalid_data(args.gtpath)
    for i in trange(len(dataset)):
        #  error_lst = ed + ey
        #  if dataset.gt_path[i].split('/')[-1][:-4] not in error_lst:
            #  continue
        try:
            x, y_reg = dataset[i]
        except:
            print('\n'*5)
            print('Error: ', dataset.gt_path[i].split('/')[-1][:-4])
        rgb = undo_normalize_rgb(x.permute(1, 2, 0).numpy())
        u_1d, d_1d = y_reg.numpy()

        plt.imshow(np.clip(rgb, 0, 1))
        plt.plot(np.arange(rgb.shape[1]), (u_1d / 2 + 0.5) * rgb.shape[0], 'b-')
        plt.plot(np.arange(rgb.shape[1]), (d_1d / 2 + 0.5) * rgb.shape[0], 'g-')
        plt.savefig('vis/dataset/%s.vis.png' % (dataset.gt_path[i].split('/')[-1][:-4]))
        plt.clf()
