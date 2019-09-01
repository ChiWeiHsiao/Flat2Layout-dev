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

