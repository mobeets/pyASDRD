import numpy as np

def woodbury(C, X, ssq):
    """
    """
    E = ssq*np.eye(X.T.shape[0]) + X.T.dot(C).dot(X)
    D = X.dot(linv(E, X.T))
    return C - C.dot(D).dot(C)
