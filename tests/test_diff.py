from aaapy import * 
import numpy as np
import scipy.linalg as la

def test_diff():
    tol = 1e-13
    X = np.linspace(-1, 1, 1000)
    F = X**2
    r = aaa(F, X)
    assert la.norm(r(X)-F, ord=np.inf) < tol
    rp = diff(r)
    assert la.norm(rp(X)-2*X, ord=np.inf) < 1e3*tol

def test_diffmat():
    tol = 1e-13
    X = np.linspace(-1, 1, 1000)
    F = X**2
    r = aaa(F, X)
    assert la.norm(r(X)-F, ord=np.inf) < tol
    D = diffmat(r.zj, r.wj)
    rp = BaryFun(r.zj, D @ r.fj, r.wj)
    assert la.norm(rp(X)-2*X, ord=np.inf) < 1e3*tol
