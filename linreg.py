import numpy as np

def solve(X, Y):
    return np.linalg.lstsq(X, Y)[0]

def predict(X, wh):
    return np.dot(X, wh)
