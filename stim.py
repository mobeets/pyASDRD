import numpy as np

class Stim:
    def __init__(self, n, nt, ns, xy=None):
        self.n = n
        self.nt = nt
        self.ns = ns
        self.S = self.stim(self.n, self.nt, self.ns)
        self.xy = randomPoints(self.ns) if xy is None else xy
        self.Sb, self.Ss, self.St = self.marginals(self.S, self.n, self.nt, self.ns)

    def marginals(self, S, n, nt, ns):
        Sb = np.reshape(S, [n, nt*ns]) # reshape
        Ss = np.sum(S, 1) # sum across time
        St = np.sum(S, 2) # sum across space
        return Sb, Ss, St

    def stim(self, n, nt, ns, mags=[2,5,8]):#, maxPulse=8):
        pulses = np.hstack([-np.array(mags[::-1]), 0, mags])
        inds = np.random.randint(len(pulses), size=[n, nt])
        St = pulses[inds]
        # St = np.random.randint(maxPulse*2+1, size=[n, nt]) - maxPulse
        
        S = np.zeros([n, nt, ns])
        for (i, j), val in np.ndenumerate(St):
            row = np.hstack([np.zeros([1, ns-abs(val)]), np.ones([1, abs(val)])])
            S[i, j, :] = (2*(val>0) - 1)*row.T[np.random.permutation(ns)].T
        return S

def randomPoints(nw, b=5, n=100):
    xi = np.linspace(-b, b, n)
    yi = xi
    idx = np.random.randint(n, size=[2, nw])
    x = xi[idx[0,:]]
    y = yi[idx[1,:]]
    return np.vstack([x, y]).T
