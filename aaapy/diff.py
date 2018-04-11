import numpy as np
from aaapy.baryfun import BaryFun

def diffmat(zj, wj):
    D = zj.reshape((-1,1)) - zj
    np.fill_diagonal(D, 1)
    D = 1/D
    D = D * (1/wj.reshape((-1,1)))
    D = D * wj
    np.fill_diagonal(D, 0)
    x = -np.ones(len(wj))
    np.fill_diagonal(D, D @ x)
    return D

def diff(r):
    D = diffmat(r.zj, r.wj)
    fj = D @ r.fj
    return BaryFun(r.zj, fj, r.wj)
