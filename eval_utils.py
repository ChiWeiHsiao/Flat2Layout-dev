import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def eval_one_CE(pc, gc, im_size):
    if len(pc) == 0 and len(gc) == 0:
        return 0
    if len(pc) == 0 or len(gc) == 0:
        return abs(len(pc) - len(gc)) / 3 / max(len(pc), len(gc))
    pc = pc.copy()
    pc[:, 0] *= im_size[1]
    pc[:, 1] *= im_size[0]
    diag = np.linalg.norm(im_size)
    distmat = cdist(pc, gc)
    row_ind, col_ind = linear_sum_assignment(distmat)  # Hungarian
    cost = distmat[row_ind, col_ind].sum()
    ptCost = cost / diag + abs(len(pc) - len(gc)) / 3
    ptCost = ptCost / max(len(pc), len(gc))
    return ptCost

def gen_1d(xys, w):
    assert len(xys) > 1
    xys = np.array(xys, np.float32)
    xys[0] = xys[1] + 1e6 * (xys[0] - xys[1])
    xys[-1] = xys[-2] + 1e6 * (xys[-1] - xys[-2])
    return np.interp(np.arange(w), xys[:, 0], xys[:, 1])

def gen_seg(c_corners, f_corners, ww_lst, shape):
    '''
    c_corners: [[x0, y0], [x1, y1], ...] telling the ceiling-wall boundary
    f_corners: same as c_cornres
    ww_lst:
        N x 2 x 2   i.e. [[[xc, yc], [xf, yf]], ...]
    '''
    seg = np.zeros(shape, np.int32)
    coorx, coory = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coorxyw = np.stack([coorx, coory, np.ones_like(coorx)], -1)

    # Generate ceiling mask
    if len(c_corners) > 1:
        c_corners = c_corners[np.argsort(c_corners[:, 0])]
        bon = gen_1d(c_corners, shape[1])
        seg[coory < bon.reshape(1, -1)] = 1

    # Generate floor mask
    if len(f_corners):
        f_corners = f_corners[np.argsort(f_corners[:, 0])]
        bon = gen_1d(f_corners, shape[1])
        seg[coory > bon.reshape(1, -1)] = 2

    # Gen walls
    wall_id = 3
    for xyc, xyf in ww_lst:
        l = np.cross([xyc[0], xyc[1], 1],
                     [xyf[0], xyf[1], 1])
        v = (coorxyw * l[None, None, :]).sum(2)
        wall_mask = ((seg == 0) & (v > 0))
        seg[wall_mask] = wall_id
        wall_id += 1

    return seg

def eval_one_PE(c_corners, f_corners, ww_lst, gt_seg, dontcare=-100, out_k=None):
    '''
    c_corners: [[x0, y0], [x1, y1], ...] telling the ceiling-wall boundary
    f_corners: same as c_cornres
    ww_lst:
        N x 2 x 2   i.e. [[[xc, yc], [xf, yf]], ...]
    '''
    pr_seg = gen_seg(c_corners, f_corners, ww_lst, gt_seg.shape)
    gt_id_space = np.unique(gt_seg)
    pr_id_space = np.unique(pr_seg)
    assert len(gt_id_space) > 0
    assert len(pr_id_space) > 0
    distmat = np.zeros((len(gt_id_space), len(pr_id_space)), np.float32)
    for gid in range(len(gt_id_space)):
        if gid == dontcare:
            continue
        for pid in range(len(pr_id_space)):
            distmat[gid, pid] = -((gt_seg == gt_id_space[gid]) & (pr_seg == pr_id_space[pid])).sum()
    row_ind, col_ind = linear_sum_assignment(distmat)  # Minimize matching loss
    overlap = -distmat[row_ind, col_ind].sum()
    pe = 1 - overlap / (gt_seg != dontcare).sum()
    if out_k:
        plt.imshow(pr_seg)
        plt.title('PE: %.4f' % pe)
        plt.savefig('vis/pr_seg/%s.png' % out_k)
        plt.clf()
    return pe


if __name__ == '__main__':
    import os
    import json
    import argparse
    from scipy.io import loadmat
    from scipy.spatial.distance import cdist
    from tqdm import trange
    import matplotlib; matplotlib.use('agg')
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--segroot', default='datas/lsun/layout_seg/')
    parser.add_argument('--gtpath', default='datas/lsun/validation.npz')
    parser.add_argument('--dtdir', required=True)
    args = parser.parse_args()

    gt = np.load(args.gtpath)
    pe_lst = []
    for ith in trange(len(gt['img_name'])):
        layout_seg_path = os.path.join(args.segroot, gt['img_name'][ith] + '.mat')
        gt_seg = loadmat(layout_seg_path)['layout']
        with open(os.path.join(args.dtdir, gt['img_name'][ith] + '.jpg.json')) as f:
            pred = json.load(f)
        c_corners = np.array(pred['cc'], np.float32).reshape(-1, 2)
        f_corners = np.array(pred['cf'], np.float32).reshape(-1, 2)
        c_all_on_edge = sum([
            ((x > 0.01) and (x < 0.99)) or ((y > 0.01) and (y < 0.99))
            for x, y in c_corners
        ]) == 0
        f_all_on_edge = sum([
            ((x > 0.01) and (x < 0.99)) or ((y > 0.01) and (y < 0.99))
            for x, y in f_corners
        ]) == 0

        c_corners[:, 0] *= gt_seg.shape[1]
        c_corners[:, 1] *= gt_seg.shape[0]
        f_corners[:, 0] *= gt_seg.shape[1]
        f_corners[:, 1] *= gt_seg.shape[0]
        ww_lst = []
        if c_all_on_edge and f_all_on_edge:
            distmat = cdist(c_corners, f_corners)
            row_ind, col_ind = linear_sum_assignment(distmat)
            ww_lst = [
                [c_corners[r], f_corners[c]]
                for r, c in zip(row_ind, col_ind)
            ]
        elif len(c_corners) == len(f_corners):
            distmat = cdist(c_corners[1:-1], f_corners[1:-1])
            row_ind, col_ind = linear_sum_assignment(distmat)
            ww_lst = [
                [c_corners[r+1], f_corners[c+1]]
                for r, c in zip(row_ind, col_ind)
            ]
        elif len(c_corners) > len(f_corners):
            distmat = cdist(c_corners[1:-1], f_corners)
            row_ind, col_ind = linear_sum_assignment(distmat)
            ww_lst = [
                [c_corners[r+1], f_corners[c]]
                for r, c in zip(row_ind, col_ind)
            ]
        else:
            distmat = cdist(c_corners, f_corners[1:-1])
            row_ind, col_ind = linear_sum_assignment(distmat)
            ww_lst = [
                [c_corners[r], f_corners[c+1]]
                for r, c in zip(row_ind, col_ind)
            ]
        pe = eval_one_PE(c_corners, f_corners, ww_lst, gt_seg, out_k=gt['img_name'][ith])
        pe_lst.append(pe)
        print(pe)

    print('Avg PE:', np.mean(pe_lst))

