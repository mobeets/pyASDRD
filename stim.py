import numpy as np
import scipy.spatial.distance

class Stim:
    def __init__(self, n, nt, ns, xy=None):
        self.n = n
        self.nt = nt
        self.ns = ns
        self.X = self.stim(self.n, self.nt, self.ns)
        self.xy = randomPoints(self.ns) if xy is None else xy
        self.D = sqdist(self.xy)
        self.Xf, self.Xs, self.Xt, self.Xm = self.marginals(self.X, self.n, self.nt, self.ns)

    def marginals(self, X, n, nt, ns):
        Xf = np.reshape(X, [n, nt*ns]) # reshape
        Xs = np.sum(X, 1) # sum across time
        Xt = np.sum(X, 2) # sum across space
        Xm = np.sum(Xf, 1) # sum across space and time
        return Xf, Xs, Xt, Xm

    def stim(self, n, nt, ns, mags=[2,5,8]):#, maxPulse=8):
        pulses = np.hstack([-np.array(mags[::-1]), 0, mags])
        inds = np.random.randint(len(pulses), size=[n, nt])
        St = pulses[inds]
        # St = np.random.randint(maxPulse*2+1, size=[n, nt]) - maxPulse
        
        X = np.zeros([n, nt, ns])
        for (i, j), val in np.ndenumerate(St):
            row = np.hstack([np.zeros([1, ns-abs(val)]), np.ones([1, abs(val)])])
            X[i, j, :] = (2*(val>0) - 1)*row.T[np.random.permutation(ns)].T
        return X

def randomPoints(nw, b=5, n=100):
    xi = np.linspace(-b, b, n)
    yi = xi
    idx = np.random.randint(n, size=[2, nw])
    x = xi[idx[0,:]]
    y = yi[idx[1,:]]
    return np.vstack([x, y]).T

def sqdist(xy):
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(xy, 'euclidean'))**2
