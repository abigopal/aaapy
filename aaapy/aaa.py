import numpy as np
import scipy.linalg as la
import scipy.sparse as sm

from aaapy.baryfun import * 

def aaa(F, Z, **kwargs):
    nind = np.isfinite(F)
    Z = Z[nind]
    F = F[nind]

    Z, uind = np.unique(Z, return_index=True)
    F = F[uind]

    mmax = kwargs.get('mmax', 100)
    tol = kwargs.get('tol', 1e-13)
    return_err = kwargs.get('return_err', False)
    cleanup = kwargs.get('cleanup', True)
    if 'ctol' not in kwargs and tol > 0:
        ctol = tol
    else:
        ctol = kwargs.get('ctol', 1e-13)

    reltol = tol * la.norm(F, ord=np.inf)
    M = len(Z)
    C = np.zeros((M, mmax), dtype=complex)
    SF = np.reshape(np.copy(F.astype(complex)), (-1,1))
    J = np.full(M, True)
    R = np.mean(F) * np.ones(M, dtype=complex)

    errvec = np.zeros(mmax)
    zj = np.zeros(mmax, dtype=complex)
    fj = np.zeros(mmax, dtype=complex)
    wj = np.zeros(mmax, dtype=complex)

    for m in range(mmax):
        err = abs(F - R)

        ix = np.argmax(err)
        zj[m] = Z[ix]
        fj[m] = F[ix]
        J[ix] = False
        C[J,m] = 1 / (Z[J] - zj[m])

        A = SF[J] * C[J,:m+1] - C[J,:m+1] * fj[:m+1]
        if len(J) < m:
            _, _, Vh = la.svd(A, full_matrices=False)
        else:
            _, _, Vh = la.svd(A)
        wj = np.conj(Vh[m,:])

        N = np.copy(F.astype(complex))
        D = np.ones(M, dtype=complex)
        N[J] = C[J,:m+1] @ (fj[:m+1] * wj)
        D[J] = C[J,:m+1] @ wj
        R = N / D

        errvec[m] = la.norm(F - R, ord=np.inf)
        if errvec[m] <= reltol:
            break

    errvec = errvec[:m+1]
    zj = zj[:m+1]
    fj = fj[:m+1]

    nz = wj != 0
    zj = zj[nz]
    fj = fj[nz]
    wj = wj[nz]

    if cleanup:
        r = BaryFun(zj, fj, wj)
        pol = r.pol()
        res = r.res()
        ix = abs(res) <= ctol * la.norm(F, ord=np.inf)
        if any(ix):
            keep = np.full(len(zj), True)
            for p in pol[ix]:
                k = np.argmin(abs(zj - p))
                keep[k] = False
                J[Z == zj[k]] = True
            zj = zj[keep]
            fj = fj[keep]
            C = 1 / (Z[J].reshape((-1,1)) - zj)
            A = SF[J] * C - C * fj
            _, _, Vh = la.svd(A)
            wj = np.conj(Vh[len(zj)-1,:])

    if return_err:
        return BaryFun(zj, fj, wj), errvec 
    else:
        return BaryFun(zj, fj, wj)
