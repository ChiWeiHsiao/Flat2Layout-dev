import os
import json
import numpy as np
from imageio import imread
from skimage.transform import resize
from scipy.ndimage.filters import maximum_filter
from scipy.signal import savgol_filter
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


def find_slope_changes(signal, window=31, percentile=95):
    # [90, 31] [95, 31]  recall 
    # [90, 63] [95, 63]  precision
    der2 = savgol_filter(signal, window_length=window, polyorder=2, deriv=2)
    max_der2 = np.max(np.abs(der2))
    cond = np.abs(der2) > np.percentile(np.abs(der2), percentile)
    # cond = np.abs(der2) > max_der2/2
    cond[signal<0] = False
    cond[signal>1] = False
    large = np.where(cond)[0]
    gaps = np.diff(large) > window
    if len(large) == 0:
        return np.array([])
    begins = np.insert(large[1:][gaps], 0, large[0])
    ends = np.append(large[:-1][gaps], large[-1])
    changes = ((begins+ends)/2).astype(np.int)
    length = ends-begins
    return changes 


def find_peaks(signal, min_v=0.05, winsz=5):
    max_v = maximum_filter(signal, size=winsz)
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    return pk_loc


def find_peaks_from_2(cor, key, bon, min_v=0.05, winsz=5):
    W = cor.shape[0]
    signal = np.max([cor, key], 0)
    sel = np.argmax([cor, key], 0)
    pk_loc = find_peaks(signal, min_v, winsz)
    #  return pk_loc, sel[pk_loc]
    x_cor = pk_loc[sel[pk_loc]==0]
    x_key = pk_loc[sel[pk_loc]==1]

    if len(pk_loc) == 0:
        # Only find turning-point from boundary when no cor/key found
        x_bon = find_slope_changes(bon, 31, 95)
        x_bon_new = []
        for i, x_tmp in enumerate(x_bon):
            if (np.abs(x_cor-x_tmp) < W*0.1).sum() == 0 and (np.abs(x_key-x_tmp) < W*0.1).sum() == 0:
                x_bon_new.append(x_bon[i])
            # else:
                # print('delete ', x_bon[i])

        x_bon = np.array(x_bon_new)
        if len(x_bon)>0:
            inside = np.logical_and(bon[x_bon]>0, bon[x_bon]<1)
            x_cor = np.concatenate([x_cor, x_bon[inside]])
            x_key = np.concatenate([x_key, x_bon[~inside]])
            cor[x_bon[inside]] = 0.2  # assign prob, to pass slope-score check
            key[x_bon[~inside]] = 0.2
            x_cor.sort()
            x_key.sort()
    return x_cor, x_key, cor, key


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
    L, slope = fit_line(xs, reg1d, mask)
    if L is None:
        return np.array([]).reshape(-1, 2)
    p0, p1 = line_X_rectangle(L)
    if p0 is None:
        return np.array([]).reshape(-1, 2)
    return np.array([p0, p1]).reshape(-1, 2)



def extract_corners(cor1d, key1d, reg1d, key_y, min_v=0.05, winsz=5, score_thresh=0.2):
    assert len(cor1d.shape) == 1
    assert len(reg1d.shape) == 1
    assert len(reg1d) % len(cor1d) == 0
    mask = (reg1d >= 0) & (reg1d <= 1)
    N = len(reg1d)
    M = len(cor1d)
    step = N // M
    #  pks = find_peaks(cor1d, min_v=min_v, winsz=winsz)
    #  pks_score = list(cor1d[pks])
    #  pks = [step//2 + v*step for v in pks]
    #  key_pks = find_peaks(key1d, min_v=min_v, winsz=winsz)
    #  key_pks_score = list(key1d[key_pks])
    #  key_pks = [step//2 + v*step for v in key_pks]

    pks, key_pks, cor1d, key1d = find_peaks_from_2(cor1d, key1d, reg1d, min_v=min_v, winsz=winsz)

    pks_score = list(cor1d[pks])
    pks = [step//2 + v*step for v in pks]
    key_pks_score = list(key1d[key_pks])
    key_pks = [step//2 + v*step for v in key_pks]
    xs = np.linspace(0, 1, N)
    

    corners = []
    keypoints = []

    pks = [0, *pks, N-1]
    PASS = False
    while not PASS:
        # Special case: no corners found
        if len(pks) == 2:  # [0, N-1]
            if len(key_pks) > 0:
                for key_x in key_pks:
                    keypoints.append(np.array([key_x/N, key_y]))
                print('='*10)
                print('Keypoint:', keypoints)
                return np.array(corners).reshape(-1,2), np.array(keypoints).reshape(-1,2)
            else:
                corners = one_line_segment_case(xs, reg1d, mask)
                return np.array(corners).reshape(-1,2), np.array(keypoints).reshape(-1,2)
     
        # Segment indices
        segments = []
        for i in range(1, len(pks)):
            segments.append([pks[i-1], pks[i]+1])

        # Find line for each segment
        Ls = []
        slopes = []
        for s, t in segments:
            L, m = fit_line(xs[s:t], reg1d[s:t], mask[s:t])
            Ls.append(L)
            slopes.append(m)

        # Check: if pks_score small & l/r slopes close -> remove peaks, merge segment
        PASS = True
        delete_idxs = []
        for i in range(len(pks_score)):
            # 0.5 = 26.57° # 0.364 = 20° # 0.268 = 15° # 0.176 = 10° # 0.0875 = 5° # 0.0349 = 2°
            fail =  not slopes[i] or not slopes[i+1] or (
                    mask[segments[i][0]:segments[i][1]].sum()>2*step and
                    mask[segments[i+1][0]:segments[i+1][1]].sum()>2*step) and (
                    (pks_score[i] < 0.5 and np.abs(slopes[i]-slopes[i+1]) < 0.176)
                    or (pks_score[i] < 0.2 and np.abs(slopes[i]-slopes[i+1]) < 0.268)
                    or (pks_score[i] < 0.1 and np.abs(slopes[i]-slopes[i+1]) < 0.364)
                    or (pks_score[i] < 0.075 and np.abs(slopes[i]-slopes[i+1]) < 0.5)
                    )
            if fail: #pks_score[i] < score_thresh and slopes[i] and slopes[i+1] and np.abs(slopes[i]-slopes[i+1]) < 0.5:
                print('Pks_score small & l/r Slopes close: ', i, pks_score[i], slopes[i], slopes[i+1])
                PASS = False
                delete_idxs.append(i)
        for idx in reversed(delete_idxs):
            pks_score.pop(idx)
            print('pks=', pks, end='\t')
            pks.pop(idx+1)
            print('-> pks=', pks)

        if PASS:
            if args.lsun_type_check and len(pks) >= 5:
                print('too many pks: ', pks)
                pks_score.pop(1)
                pks.pop(2)
                print(' -> ', pks)
                PASS = False
                print('='*100)

    assert(len(pks_score) == len(pks)-2)
    assert(pks[0]==0 and pks[-1]==N-1)

    # Connect line into corners
    if len(Ls) == 2 and (Ls[0] is None or Ls[1] is None):
        print('Fall back: two segments only but at least one dont have line')
        print(segments[0][0], reg1d[segments[0][0]])
        print(segments[1][0], reg1d[segments[1][0]])
        cor_lr = one_line_segment_case(xs, reg1d, mask)
        cor_mid = np.array([[pks[1]/N, np.clip(reg1d[pks[1]], 0, 1)]])
        if len(cor_lr) > 0:
            #  return np.vstack([cor_lr[[0]], cor_mid, cor_lr[[1]]])
            corners = np.vstack([cor_lr[[0]], cor_mid, cor_lr[[1]]])
            return np.array(corners).reshape(-1,2), np.array(keypoints).reshape(-1,2)
        else:
            # check corner peak confidence: if small->only 1, if large->repeat 3
            if pks_score[0] < score_thresh:
                #  return cor_mid
                corners = cor_mid
                return np.array(corners).reshape(-1,2), np.array(keypoints).reshape(-1,2)
            else:
                #  return np.vstack([cor_mid, cor_mid, cor_mid])
                corners = np.vstack([cor_mid, cor_mid, cor_mid])
                return np.array(corners).reshape(-1,2), np.array(keypoints).reshape(-1,2)

    for i in range(1, len(Ls)-1):
        if Ls[i] is None:
            print(Ls[i])
        assert Ls[i] is not None, 'Middle segments dont have line !??'


    corners = []
    if Ls[0] is None:
        p0, p1 = line_X_rectangle(Ls[1])
        corners.append(p0)
        if pks_score[0] > score_thresh:
            corners.append(p0)
        #  raise ValueError('Ls[0] == None')
    else:
        p0, p1 = line_X_rectangle(Ls[0])
        corners.append(p0)
    
    for i in range(1, len(Ls)):
        if Ls[i-1] is None or Ls[i] is None:
            continue
        pts = np.cross(Ls[i-1], Ls[i])
        #  corners.append(pts[:2] / pts[2])

        # If intersection x is too far from predicted y_cor, xy=[y_cor, y_reg[y_cor]]
        toofar = 16
        pts = pts[:2] / pts[2]  # [x,y,z]->[x/z,y/z]
        if np.abs(N*pts[0] - pks[i]) < toofar:
            corners.append(pts[:2])
        else:
            pred_x = pks[i]
            pred_y = min(max(reg1d[pred_x], 0), 1)
            corners.append(np.array([pred_x/N, pred_y]))
    
    if Ls[-1] is None:
        p0, p1 = line_X_rectangle(Ls[-2])
        corners.append(p1)
        if pks_score[-1] > score_thresh:
            corners.append(p1)
    else:
        p0, p1 = line_X_rectangle(Ls[-1])
        corners.append(p1)

    # Check failure
    for c in corners:
        if c[0] is None or c[0] < -0.1 or c[0] > 1.1 or c[1] < -0.1 or c[1] > 1.1:
            print(c)
            print('corners: ', corners)
            print('Failure: c[0] is None or c[0] < -0.1 or c[0] > 1.1 or c[1] < -0.1 or c[1] > 1.1')
            #  return one_line_segment_case(xs, reg1d, mask)
            corners = one_line_segment_case(xs, reg1d, mask)
            return np.array(corners).reshape(-1,2), np.array(keypoints).reshape(-1,2)

    #  return np.array(corners).reshape(-1, 2)
    return np.array(corners).reshape(-1,2), np.array(keypoints).reshape(-1,2)


def check_lsun_type(c1s, k1s, bon1, c2s, k2s, bon2):
    ns = [len(c1s), len(k1s), len(c2s), len(k2s)]
    valid_ns = [[4,0,4,0], [0,2,4,0], [4,0,0,2], [3,0,0,1], [0,1,3,0], [3,0,3,0], [2,0,2,0], [0,2,0,2], [2,0,0,0], [0,0,2,0], [0,1,0,1]]
    if ns not in valid_ns:
        assert('not valid type')


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
    parser.add_argument('--min_v', default=0.2, type=float)
    parser.add_argument('--winsz', default=32, type=int)
    parser.add_argument('--lsun_type_check', action='store_true')
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
    not_valid = []

    for ith in trange(len(gt['img_name'])):
        # filter_lst = ['sun_amuqfplrtismkqxi', 'sun_bnueorjiyqdbwyon', 'sun_aiissweorvnwfbhi', 'sun_afnwwltwbnosxqeq',
                # 'sun_ajtnlavhnmemhbhs', 'sun_apdausmjeciamxir', 'c944b29b38bea850a64ab50d5bb3451ea8bcd97e',
                # 'sun_abvkcirmhtntrxmr', 'sun_abvkcirmhtntrxmr', 'sun_azovcesifwbjywhw',
                # 'sun_agtgmkmqzawiktsb', 'sun_bwlhyzdsslllylqc', '196c74152416e5cd8fa8787fdc0b3ec9cc38274b',
                # 'sun_astvrdmzjytrlgyi', '4fe8887c6204a0bb59116996b2100af2e1e6d317',
                # '3850db9fbf340a960f9714053ab08b35a354c38e']
        # filter_lst = ['sun_blhukmnfxukrwagk']
        # if gt['img_name'][ith] not in filter_lst:
            # continue
        # print('='*20)
        print(gt['img_name'][ith])

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
            output = net(x[None])
            out_reg, out_cor, out_key = output
            # Upsample W/32 --> W
            y_step = w // out_reg.shape[-1]
            pad_left = (2*out_reg[...,[0]]-out_reg[...,[1]])
            pad_right = (2*out_reg[...,[-1]]-out_reg[...,[-2]])
            out_reg = torch.cat([pad_left, out_reg, pad_right], -1)
            out_reg = F.interpolate(out_reg, scale_factor=y_step, mode='linear', align_corners=False)
            out_reg = out_reg[..., y_step:-y_step].clamp(min=-1.05, max=1.05)
            #  out_reg = F.interpolate(out_reg, size=w, mode='linear', align_corners=False)
            np_reg = out_reg[0].cpu().numpy() / 2 + 0.5  # [-1, 1] => [0, 1]
            np_cor = torch.sigmoid(out_cor[0]).cpu().numpy()
            np_key = torch.sigmoid(out_key[0]).cpu().numpy()

        corners_c, keypoints_c = extract_corners(np_cor[0], np_key[0], np_reg[0], key_y=0, min_v=args.min_v, winsz=args.winsz)
        corners_f, keypoints_f = extract_corners(np_cor[1], np_key[1], np_reg[1], key_y=1, min_v=args.min_v, winsz=args.winsz)

        # Different sides, add corner based on keypoint's x
        near_thresh = 64/w
        for key in keypoints_c:
            tmp_x = key[0]
            if (np.abs(corners_f[:, 0]-tmp_x) < near_thresh).sum() == 0:
                tmp = np.array([[tmp_x, min(1, np_reg[1, int(round(w*tmp_x))])]])
                corners_f = np.vstack([corners_f, tmp])
        for key in keypoints_f:
            tmp_x = key[0]
            if (np.abs(corners_c[:, 0]-tmp_x) < near_thresh).sum() == 0:
                tmp = np.array([[tmp_x, max(0, np_reg[0, int(round(w*tmp_x))])]])
                corners_c = np.vstack([corners_c, tmp])


        # only for LSUN: too more corners
        if args.lsun_type_check:
            check_lsun_type(corners_c, keypoints_c, np_reg[0], corners_f, keypoints_f, np_reg[1])


        # merge corners and keypoints
        corners_c = np.vstack([corners_c, keypoints_c])
        corners_f = np.vstack([corners_f, keypoints_f])
        # 
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
        # print('Ceiling ', np.array([w, h])*corners_c)
        # print('Floor   ', np.array([w, h])*corners_f)


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
            u_cor, d_cor = np_cor.repeat(w/np_cor.shape[-1], axis=-1)
            u_cor = u_cor.reshape([1,-1]).repeat(10, axis=0)
            d_cor = d_cor.reshape([1,-1]).repeat(10, axis=0)
            u_cor = u_cor.reshape([10, w, 1]).repeat(3, axis=2)
            d_cor = d_cor.reshape([10, w, 1]).repeat(3, axis=2)
            u_cor[..., [0,2]] = 0  # keep only G
            d_cor[..., [0,1]] = 0  # keep only B
            rgb = np.vstack([rgb, u_cor, d_cor])

            u_key, d_key = np_key.repeat(w/np_key.shape[-1], axis=-1)
            u_key = u_key.reshape([1,-1]).repeat(10, axis=0)
            d_key = d_key.reshape([1,-1]).repeat(10, axis=0)
            u_key = u_key.reshape([10, w, 1]).repeat(3, axis=2)
            d_key = d_key.reshape([10, w, 1]).repeat(3, axis=2)
            u_key[..., [2]] = 0  # R+G
            d_key[..., [1]] = 0  # R+B
            rgb = np.vstack([rgb, u_key, d_key])

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
