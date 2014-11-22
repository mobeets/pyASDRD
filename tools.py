import numpy as np
import scipy.linalg
import timeit

def woodbury(C, X, ssq):
    """
    """
    E = ssq*np.eye(X.T.shape[0]) + X.T.dot(C).dot(X)
    D = X.dot(linv(E, X.T))
    return C - C.dot(D).dot(C)

def linv(A, y):
    """
    inv(A)*y
    """
    return np.linalg.solve(A, y)

def rinv(A, y):
    """
    y*inv(A)
    """
    return np.linalg.solve(A.T, y.T).T

def PostCov(RegInv, XX, ssq):
    return np.linalg.inv((XX/ssq) + RegInv)

def PostMean(sigma, XY, ssq):
    return sigma.dot(XY)/ssq
