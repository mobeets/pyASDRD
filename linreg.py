import numpy as np

def ols(X, Y):
    return np.linalg.lstsq(X, Y)[0]

def bilinear(X, Y, niters=1000):
    whs = solve(np.sum(X, 1), Y)
    for _ in xrange(niters):
        wht = solve(X.dot(whs), Y)
        whs = solve(wht.dot(X), Y)
    return whs, wht
