import numpy as np
import scipy.linalg as la

class BaryFun:
    m = 0
    zj = np.array([], dtype=complex)
    fj = np.array([], dtype=complex)
    wj = np.array([], dtype=complex)
    _pol = np.array([], dtype=complex)
    _res = np.array([], dtype=complex)
    _zer = np.array([], dtype=complex)
    
    def __init__(self, zj, fj, wj):
        m = len(zj)

        if len(fj) != m or len(wj) != m:
            raise ValueError('Inputs must be same length.')

        self.m = m
        self.zj = zj
        self.fj = fj
        self.wj = wj
    
    def pol(self):
        if len(self._pol) == 0:
            m = self.m
            zj = self.zj
            wj = self.wj
            B = np.identity(m+1, dtype=complex)
            B[0,0] = 0
            A = np.zeros((m+1, m+1), dtype=complex)
            A[0,1:] = wj
            A[1:,0] = 1
            np.fill_diagonal(A[1:,1:], zj)
            pol,_ = la.eig(A,B)
            self._pol = pol[~np.isinf(pol)]
        return self._pol

    def zer(self):
        if len(self._zer) == 0:
            m = self.m
            zj = self.zj
            fj = self.fj
            wj = self.wj
            B = np.identity(m+1, dtype=complex)
            B[0,0] = 0
            A = np.zeros((m+1, m+1), dtype=complex)
            A[0,1:] = wj * fj
            A[1:,0] = 1
            np.fill_diagonal(A[1:,1:], zj)
            zer,_ = la.eig(A,B)
            self._zer = zer[~np.isinf(zer)]
        return self._zer

    def res(self):
        if len(self._res) == 0:
            m = self.m
            zj = self.zj
            fj = self.fj
            wj = self.wj
            pol = self.pol()
            C = np.zeros((len(pol), m), dtype=complex)
            C = C + pol.reshape((-1,1))
            C = C - zj
            C = 1/C
            N = C @ (wj*fj)
            Ddiff = -C**2 @ wj
            self._res = N/Ddiff
        return self._res

    def __call__(self, Z):
        m = self.m
        zj = self.zj
        fj = self.fj
        wj = self.wj

        scalar = np.isscalar(Z)
        if scalar:
            Z = np.array([Z])

        M = len(Z)
        N = np.ones(M, dtype=complex)
        D = np.ones(M, dtype=complex)

        keep = np.full(M, True)
        for k in range(m):
            ind = (Z == zj[k])
            N[ind] = fj[k]
            keep = keep & ~ind

        C = np.zeros((sum(keep), m), dtype=complex)
        C = C + Z[keep].reshape((-1,1))
        C = C - zj

        C = 1/C
        D[keep] = C @ wj
        N[keep] = C @ (wj*fj)
        R = N / D

        if scalar:
            R = R[0]

        return R
