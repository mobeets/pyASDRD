import numpy as np
import scipy.spatial.distance

class Stim(object):
    def __init__(self, X, xy):
        self.n, self.nt, self.ns = X.shape
        self.X = X
        self.xy = xy
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

class RandStim(Stim):
    def __init__(self, n, nt, ns, xy=None):
        self.X = self.stim(n, nt, ns)
        self.xy = griddedPoints(ns) if xy is None else xy
        super(RandStim, self).__init__(self.X, self.xy)

def randomPoints(nw, b=5, n=100):
    xi = np.linspace(-b, b, n)
    yi = xi
    idx = np.random.randint(n, size=[2, nw])
    x = xi[idx[0,:]]
    y = yi[idx[1,:]]
    return np.vstack([x, y]).T

def griddedPoints(nw, b=5, hexLike=True):
    assert np.round(np.sqrt(nw))**2 == nw
    nx = np.round(np.sqrt(nw))
    ny = nw*1.0/nx
    xi = np.linspace(-b, b, nx)
    yi = np.linspace(-b, b, ny)
    zi = np.meshgrid(xi, yi)
    if hexLike:
        zi[0][0::2,:] += np.diff(xi).min()/2.0
    return np.array(zip(zi[0].flatten(), zi[1].flatten()))

def sqdist(xy):
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(xy, 'euclidean'))**2

def Dt(nt, ns):
    xy = np.array(zip(np.arange(nt), np.zeros(nt)))
    D = sqdist(xy)
    z = np.zeros([ns, ns])
    return np.hstack([np.vstack([z+i for i in D[:,j]]) for j in xrange(nt)])
