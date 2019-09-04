import os
import json
import numpy as np
from imageio import imread
from skimage.transform import resize
from scipy.ndimage.filters import maximum_filter
from scipy.stats import siegelslopes
from cohenSutherlandClip import cohenSutherlandClip
from tqdm import trange
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.nn.functional as F

from init import init
from utils import load_trained_model
from dataset import normalize_rgb
import eval_utils


def find_peaks(signal, winsz=3, min_v=0.05):
    max_v = maximum_filter(signal, size=winsz)
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    return pk_loc


def fit_line(xs, ys, mask):
    if len(xs) == 0:
        return None, None
    if mask.sum() < 2:
        return None, None
    xs = xs[mask]
    ys = ys[mask]
    m, b = siegelslopes(ys, xs)
    # line in homogeneous coordinate
    L = np.cross([0, 0*m+b, 1], [1, 1*m+b, 1])
    return L, m


def interp_L(L, x=None, y=None):
    if x is None:
        return (-L[1]*y - L[2]) / L[0]
    else:
        return (-L[0]*x - L[2]) / L[1]


def line_X_rectangle(L):
    p0 = (-0.1, interp_L(L, x=-0.1))
    p1 = (1.1, interp_L(L, x=1.1))
    p0, p1 = cohenSutherlandClip(p0, p1)
    return p0, p1


def one_line_segment_case(xs, reg1d, mask):
    L, m = fit_line(xs, reg1d, mask)
    if L is None:
        return np.array([]).reshape(-1, 2)
    p0, p1 = line_X_rectangle(L)
    if p0 is None:
        return np.array([]).reshape(-1, 2)
    return np.array([p0, p1]).reshape(-1, 2)


def extract_corners(cor1d, reg1d, min_v=0.05):
    assert len(cor1d.shape) == 1
    assert len(reg1d.shape) == 1
    assert len(reg1d) % len(cor1d) == 0
    mask = (reg1d >= 0) & (reg1d <= 1)
    N = len(reg1d)
    M = len(cor1d)
    step = N // M
    pks = [step//2 + v*step for v in find_peaks(cor1d, min_v=min_v)]
    xs = np.linspace(0, 1, N)

    # Special case: no corners found
    if len(pks) == 0:
        return one_line_segment_case(xs, reg1d, mask)
 
    # Segment indices
    pks = [0, *pks, N-1]
    segments = []
    for i in range(1, len(pks)):
        segments.append([pks[i-1], pks[i]+1])

    # Find line for each segment
    Ls = []
    ms = []
    for s, t in segments:
        L, m = fit_line(xs[s:t], reg1d[s:t], mask[s:t])
        #  if np.abs(m-ms[-1]) < 0.5:
        Ls.append(L)
        ms.append(m)

    # Connect line into corners
    if len(Ls) == 2 and (Ls[0] is None or Ls[1] is None):
        print('Fall back: two segments only but at least one dont have line')
        print(segments[0][0], reg1d[segments[0][0]])
        print(segments[1][0], reg1d[segments[1][0]])
        #  if np.logical_and(reg1d[mask]>=0, reg1d[mask]<=1).sum() > 0:
            #  raise ValueError('Fall back: two segments only but at least one dont have line')
        return one_line_segment_case(xs, reg1d, mask)
    for i in range(1, len(Ls)-1):
        return one_line_segment_case(xs, reg1d, mask)
        #  assert Ls[i] is not None, 'Middle segments dont have line !??'
    corners = []
    if Ls[0] is None:
        p0, p1 = line_X_rectangle(Ls[1])
        corners.append(p0)
        corners.append(p0)
        #  raise ValueError('Ls[0] == None')
    else:
        p0, p1 = line_X_rectangle(Ls[0])
        corners.append(p0)
    
    for i in range(1, len(Ls)):
        if Ls[i-1] is None or Ls[i] is None:
            #  print('i: ', i)
            #  print('peak corners: ', pks)
            #  print('Ls[i-1]: ', Ls[i-1])
            #  print('Ls[i]: ', Ls[i])
            #  print(segments[i][0], reg1d[segments[i][0]])
            #  print(segments[i-1][0], reg1d[segments[i-1][0]])
            #  raise ValueError('Ls[i-1] is None or Ls[i] is None')
            continue
        pts = np.cross(Ls[i-1], Ls[i])
        #  corners.append(pts[:2] / pts[2])
        # If intersection x is too far from predicted y_cor, xy=[y_cor, y_reg[y_cor]]
        toofar = 16
        if np.abs(N*pts[0] - pks[i]) < toofar:
            corners.append(pts[:2] / pts[2])
        else:
            pred_x = pks[i]
            #  print('intersection too far')
            #  print('intersect:', pts[:2] / pts[2] * N)
            #  print('new      :', [pred_x, reg1d[pred_x]*N])
            corners.append(np.array([pred_x/N, reg1d[pred_x]]))
    
    if Ls[-1] is None:
        p0, p1 = line_X_rectangle(Ls[-2])
        corners.append(p1)
        corners.append(p1)
        #  raise ValueError('Ls[-1] == None')
    else:
        p0, p1 = line_X_rectangle(Ls[-1])
        corners.append(p1)

    # Check failure
    for c in corners:
        if c[0] is None or c[0] < -0.1 or c[0] > 1.1 or c[1] < -0.1 or c[1] > 1.1:
            return one_line_segment_case(xs, reg1d, mask)

    #  # Check corners if too close, delete one
    #  corners = np.array(corners).reshape(-1, 2)
    #  nC = corners.shape[0]
    #  dist = cdist(corners, corners)
    #  dist = dist + 1e9*np.eye(nC)
    #  amin = dist.argmin(0)
    #  tooclose = dist[np.arange(nC), amin] < 0.016
    #  if tooclose.sum():
        #  print('too close, dist=: ', dist[tooclose, amin[tooclose]])
        #  print('\nori corners:', corners*640)
        #  for i in range(nC):
            #  if tooclose[i]:
                #  print('Pair ', i, 640*corners[i], 640*corners[amin[i]])
                #  tmp_x, tmp_y = corners[i]
                #  if tmp_x==0 or tmp_x==1 or tmp_y==0 or tmp_y==1:
                    #  tooclose[i] = False
        #  corners = corners[~tooclose]
        #  print('\nnew corners:', corners*640)
    return np.array(corners).reshape(-1, 2)



if __name__ == '__main__':

    import argparse
    import matplotlib; matplotlib.use('agg')
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--imgroot', default='datas/lsun/images')
    parser.add_argument('--gtpath', default='datas/lsun/validation.npz')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--min_v', default=0.05, type=float)
    parser.add_argument('--debug')
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    net, kwargs = load_trained_model(args.pth)
    net = net.to(device)
    net.eval()
    print(kwargs)

    h = kwargs['main_h']
    w = kwargs['main_w']
    gt = np.load(args.gtpath)
    all_ce = []

    for ith in trange(len(gt['img_name'])):
        path = os.path.join(args.imgroot, gt['img_name'][ith])
        print(path)
        if args.debug and os.path.split(path)[1] != args.debug:
            continue
        if os.path.isfile(path):
            pass
        elif os.path.isfile(path + '.jpg'):
            path = path + '.jpg'
        elif os.path.isfile(path + '.png'):
            path = path + '.png'
        else:
            raise Exception('%s found !??' % path)

        gt_corner_c = np.array(gt['ceiling_corner'][ith]).reshape(-1, 2)
        gt_corner_f = np.array(gt['floor_corner'][ith]).reshape(-1, 2)
        gt_corner_cw = np.array(gt['ceiling_wall'][ith]).reshape(-1, 2)
        gt_corner_fw = np.array(gt['floor_wall'][ith]).reshape(-1, 2)
        if len(gt_corner_c) == 0:
            gt_corner_c = gt_corner_cw
        if len(gt_corner_f) == 0:
            gt_corner_f = gt_corner_fw

        # Prepare input
        rgb = imread(path)[..., :3] / 255
        ori_h, ori_w = rgb.shape[:2]
        if (ori_h, ori_w) != (h, w):
            rgb = resize(rgb, [h, w], preserve_range=True, anti_aliasing=False, mode='reflect')
        x = torch.FloatTensor(normalize_rgb(rgb).transpose(2, 0, 1).copy()).to(device)

        # Prepare output
        with torch.no_grad():
            out_reg, out_cor = net(x[None])
            out_reg = F.interpolate(out_reg, size=w, mode='linear', align_corners=False)
            np_reg = out_reg[0].cpu().numpy() / 2 + 0.5  # [-1, 1] => [0, 1]
            np_cor = torch.sigmoid(out_cor[0]).cpu().numpy()

        corners_c = extract_corners(np_cor[0], np_reg[0], min_v=args.min_v)
        corners_f = extract_corners(np_cor[1], np_reg[1], min_v=args.min_v)
        if len(corners_c) != 0 and len(corners_f) == 0:
            corners_f = np.array([
                [x, 1]
                for x, y in corners_c[1:-1]
            ]).reshape(-1, 2)
        elif len(corners_c) == 0 and len(corners_f) != 0:
            corners_c = np.array([
                [x, 0]
                for x, y in corners_f[1:-1]
            ]).reshape(-1, 2)

        try:
            ce = eval_utils.eval_one_CE(
                np.concatenate([corners_c, corners_f], 0),
                np.concatenate([gt_corner_c, gt_corner_f], 0),
                [ori_h, ori_w])
        except:
            ce = eval_utils.eval_one_CE(
                    np.array([]).reshape(-1,2),
                    np.concatenate([gt_corner_c, gt_corner_f], 0),
                    [ori_h, ori_w])
        all_ce.append(ce)

        with open('%s/%s.json' % (args.out, os.path.split(path)[1]), 'w') as f:
            json.dump({
                'cc': corners_c.tolist(),
                'cf': corners_f.tolist(),
            }, f)

        if args.vis:
            u_cor, d_cor = np_cor.repeat(32, axis=-1)
            u_cor = u_cor.reshape([1,-1]).repeat(10, axis=0)
            d_cor = d_cor.reshape([1,-1]).repeat(10, axis=0)
            u_cor = u_cor.reshape([10, w, 1]).repeat(3, axis=2)
            d_cor = d_cor.reshape([10, w, 1]).repeat(3, axis=2)
            u_cor[..., [0,2]] = 0  # keep only G
            d_cor[..., [0,1]] = 0  # keep only B
            rgb = np.vstack([rgb, u_cor, d_cor])
            plt.imshow(np.clip(rgb, 0, 1))
            plt.plot(np.linspace(0, 1, w) * w, np_reg[0] * h, 'g')
            plt.plot(np.linspace(0, 1, w) * w, np_reg[1] * h, 'b')
            if corners_c is not None:
                if len(corners_c) and sum([v is None or v[0] is None or v[1] is None for v in corners_c]) == 0:
                    plt.plot([x * w for x, y in corners_c], [y * h for x, y in corners_c], 'ro-')
            if corners_f is not None:
                if len(corners_f) and sum([v is None or v[0] is None or v[1] is None for v in corners_f]) == 0:
                    plt.plot([x * w for x, y in corners_f], [y * h for x, y in corners_f], 'ro-')        
            if len(gt_corner_c):
                plt.plot([x * w / ori_w for x, y in gt_corner_c], [y * h / ori_h for x, y in gt_corner_c], 'yo-')
            if len(gt_corner_f):
                plt.plot([x * w / ori_w for x, y in gt_corner_f], [y * h / ori_h for x, y in gt_corner_f], 'yo-')
            plt.xlim(0, rgb.shape[1])
            plt.ylim(rgb.shape[0], 0)
            plt.savefig('%s/%.4f_%s.png' % (args.out, ce, os.path.split(path)[1]))
            plt.clf()


    print('Avg CE: %.4f' % np.mean(all_ce))

