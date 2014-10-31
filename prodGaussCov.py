import numpy as np

def linv(A, y):
    """
    inv(A)*y
    """
    return np.linalg.lstsq(A, y)[0]

def rinv(A, y):
    """
    y*inv(A)
    """
    return np.linalg.lstsq(A.T, y.T)[0].T

def prodGaussCov(C, X, ssq):
    """
    """
    E = ssq*np.eye(X.T.shape[0]) + X.T.dot(C).dot(X)
    # D = X.dot(np.linalg.inv(E)).dot(X.T)
    D = X.dot(linv(E, X.T))
    return C - C.dot(D).dot(C)

if __name__ == '__main__':
    n = 7
    X = np.random.random([n+2, n])
    A = X.T.dot(X)
    C = np.random.random([n, n])
    ssq = np.random.random()
    Y0 = np.linalg.inv(A/ssq + np.linalg.inv(C))
    Y = prodGaussCov(C, X.T, ssq)
    assert np.allclose(Y0, Y)
