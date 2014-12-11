import argparse
import numpy as np

def split(X, Y, N, M, startM=0, front=True):
    """
    N is int - number of trials
    M is int - number of lags
    """
    if front:
        X = X[:N, startM:startM+M]
        Y = Y[:N]
    else:
        X = X[-(N+1):, startM:startM+M]
        Y = Y[-(N+1):]
    return X, Y

def load_raw(infile, keep_ones=True):
    assert infile.endswith('.npy')
    xs = np.load(infile)
    end_ind = -1 if keep_ones else -2
    X = xs[:,:end_ind][:,::-1] # all but last columns; reverse columns
    Y = xs[:,-1] # last column
    return X, Y

def load(infile, nLags, trainPct):
    X, Y = load_raw(infile, keep_ones=False)
    if p == 0.0:
        N = X.shape[0]
    elif p > 0.0:
        N = int(p*X.shape[0])
    if nLags == 0:
        nLags = X.shape[1]
    X0, Y0 = split(X, Y, N, nLags, skipM, front=True)
    X1, Y1 = split(X, Y, X.shape[0] - N, nLags, skipM, front=False)
    D = temporal_distance(X0.shape[1])
    return X0, Y0, X1, Y1, D

def PostCov(RegInv, XX, ssq):
    return np.linalg.inv((XX/ssq) + RegInv)

def PostMean(sigma, XY, ssq):
    return sigma.dot(XY)/ssq

logDet = lambda x: np.linalg.slogdet(x)[1]
def ASDEvi(X, Y, Reg, sigma, ssq, p, q):
    """
    log evidence
    """
    z1 = 2*np.pi*sigma
    z2 = 2*np.pi*ssq*np.eye(q)
    z3 = 2*np.pi*Reg
    logZ = logDet(z1) - logDet(z2) - logDet(z3)
    B = (np.eye(p)/ssq) - (X.dot(sigma).dot(X.T))/(ssq**2)
    return 0.5*(logZ - Y.T.dot(B).dot(Y))

def main(infile, trainPct, nLags, ro, ssq, delta):
    load(infile, nLags, trainPct)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, default=None)
    parser.add_argument('--ro', type=float, default=None)
    parser.add_argument('--ssq', type=float, default=None)
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('-p', type=float, default=0.0, help="percent of data to use in training")
    parser.add_argument('-m', type=int, default=0, help="# of lags to use")
    args = parser.parse_args()
    main(args.infile, args.p, args.m, args.ro, args.ssq, args.delta)
