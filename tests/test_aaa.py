from aaapy import aaa
import numpy as np
from scipy.special import gamma
import scipy.linalg as la
import warnings

def test_basics():
    tol = 1e-10
    Z = np.linspace(-1, 1, 1e3)
    F = np.exp(Z)
    r = aaa(F, Z)
    assert np.isnan(r(np.nan))
    m1 = r.m
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = aaa(F, Z, mmax=m1-1)
    assert r.m == m1-1
    r = aaa(F, Z, tol=1e-3)
    assert r.m < m1

def test_tan():
    tol = 1e-10
    Z = np.linspace(-1, 1, 1e3)
    F = np.tan(np.pi * Z)
    r = aaa(F, Z)
    assert max(abs(F - r(Z))) < tol
    assert min(abs(r.zer())) < tol
    assert min(abs(r.pol() - 0.5)) < tol
    assert min(abs(r.res())) > 1e-13

def test_scaleinvar():
    Z = np.linspace(0.3, 1.5, 1e2)
    F = np.exp(Z)/(1+1j)
    r1 = aaa(F, Z)
    r2 = aaa(2**30 * F, Z)
    r3 = aaa(2**30 * F, Z)
    assert r1(0.2j) == 2**-30 * r2(0.2j)
    assert r1(1.4) == 2**-30 * r3(1.4)

def test_inf():
    tol = 1e-3
    Z = np.linspace(-1, 1, 1e2)
    F = gamma(Z)
    r = aaa(F, Z)
    assert abs(r(0.63) - gamma(0.63)) < tol

def test_nan():
    tol = 1e-3
    Z = np.linspace(0, 20, 1e2)
    np.seterr(all='ignore')
    F = np.sin(Z) / Z
    np.seterr(all=None)
    r = aaa(F, Z)
    assert abs(r(2) - np.sin(2)/2) < tol
