import numpy as np
import scipy.linalg
import timeit

def woodbury(C, X, ssq):
    """
    """
    E = ssq*np.eye(X.T.shape[0]) + X.T.dot(C).dot(X)
    D = X.dot(linv(E, X.T))
    return C - C.dot(D).dot(C)

def linv2(A, y):
    return getattr(y, "__array_prepare__", y.__array_wrap__)(np.linalg._umath_linalg.solve(A, y, signature='dd->d').astype(np.double))

def linv(A, y):
    """
    inv(A)*y
    """
    return np.linalg.solve(A, y)
    # return np.linalg.lstsq(A, y)[0]

def rinv(A, y):
    """
    y*inv(A)
    """
    return np.linalg.solve(A.T, y.T).T
    # return np.linalg.lstsq(A.T, y.T)[0].T

# also see here: http://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy

# def mylstsq(x, y):
#     " https://www.marshut.net/wrwws/least-squares-speed.html "
#     q, r = np.linalg.qr(x, 'reduced')
#     y = np.dot(q.T, y)
#     return np.linalg.lstsq(r, y)

def setup():
    n = 100
    X = np.random.random([n, n*10])
    C = np.random.random([n, n])
    E = np.eye(X.T.shape[0]) + X.T.dot(C).dot(X)
    return E, X

def test_woodbury(N=1000):
    x0 = "from __main__ import setup, linv2; from tmpsolve import solve; import scipy.linalg; import numpy as np; E, X = setup();"
    x1 = "scipy.linalg.solve(E, X.T)"
    x2 = "linv2(E, X.T)"
    x3 = "np.linalg.solve(E, X.T)"
    print
    print timeit.timeit(x1, setup=x0, number=N)
    print
    print timeit.timeit(x2, setup=x0, number=N)
    print
    print timeit.timeit(x3, setup=x0, number=N)
    print

if __name__ == '__main__':
    test_woodbury()
