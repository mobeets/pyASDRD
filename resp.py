import scipy.stats
import numpy as np

class Resp:
    def __init__(self, S, signalType, ssq=0.0, SNR=None, wt=None, ws=None, wf=None):
        """
        S.X is space-time stimulus on each trial
        S.xy is x,y locations of space as represented in stimulus
        ssq is variance of noise in response; ignored if SNR is provided
        SNR is desired ratio of signal variance to noise variance
        wt, ws, wf are the time, space, or full weights, respectively
        signalType in ['bilinear', 'spacey', 'full', 'rank-k'] where k is int
        
        calculates the weighted response to the given stimulus
        """
        self.SNR = SNR
        self.ssq = ssq if self.SNR is not None else None
        (n, nt, ns) = S.X.shape
        self.signalType = signalType
        if self.signalType == 'full':
            self.wf = randomFullRank(nt, ns)
            self.wt = None
            self.ws = None
            self.sig_fcn = self.full_signal
        elif self.signalType == 'spacey':
            self.ws = 10*randomGaussianWeights(S.xy)
            self.wt = None
            self.wf = None
            self.sig_fcn = self.spacey_signal
        elif self.signalType == 'bilinear':
            self.wt, self.ws = randomBilinear(nt, S.xy, norm=True)
            self.wf = None
            self.sig_fcn = self.bilinear_signal
        elif self.signalType.startswith('rank-'):
            k = int(self.signalType.split('rank-')[1])
            self.wf = randomRankK(nt, ns, S.xy, k=k)
            self.wt = None
            self.ws = None
            self.sig_fcn = self.full_signal
        elif ws is not None:
            self.ws = ws
            self.wf = None
            if wt is not None:
                self.wt = wt
                self.sig_fcn = self.bilinear_signal
            else:
                self.sig_fcn = self.spacey_signal
                self.wt = None
        elif wf is not None:
            self.wf = wf
            self.wt = None
            self.ws = None
            self.sig_fcn = self.full_signal
        self.sig_fcn_lkp = {'bilinear': self.bilinear_signal, 'spacey': self.spacey_signal, 'full': self.full_signal}
        self.sig_fcn = self.sig_fcn_lkp.get(self.signalType, self.full_signal)
        self.Y = self.resp(S.X, self.wf, self.wt, self.ws)

    def full_signal(self, X, wf=None, wt=None, ws=None):
        return np.einsum('abc,bc -> a', X, wf)

    def bilinear_signal(self, X, wf=None, wt=None, ws=None):
        return wt.dot(X).dot(ws)

    def spacey_signal(self, X, wf=None, wt=None, ws=None):
        return np.sum(X, 1).dot(ws)

    def resp(self, X, wf, wt, ws):
        (n, nt, ns) = X.shape
        self.Ysig = self.sig_fcn(X, wf, wt, ws) # signal
        if self.SNR is not None:
            self.ssq = np.var(self.Ysig)/self.SNR
        self.Ynse = np.random.normal(0, np.sqrt(self.ssq), n) if self.ssq > 0 else 0 # noise
        return self.Ysig + self.Ynse

def randomFullRank(nt, ns):
    return np.random.rand(nt, ns)

def randomBilinear(nt, xy, norm=True):
    wt = randomTimeWeights(nt)
    ws = randomGaussianWeights(xy)
    if norm:
        # scale so space and time roughly equal
        ws = ws*(np.sum(wt)/np.sum(ws))
    return wt, ws

def randomRankK(nt, ns, xy, k=1):
    U = np.zeros([nt, k])
    V = np.zeros([ns, k])
    for i in xrange(k):
        U[:,i], V[:,i] = randomBilinear(nt, xy, norm=True)
    U, V = U/U.sum(), V/V.sum()
    S = np.diag(2*np.random.random(k))
    return U.dot(S).dot(V.T)

def randomTimeWeights(nt):
    k = 4*np.random.random()+1
    th = np.random.random()+1
    x = np.arange(nt)
    wt_fcn = lambda k, th: (x**(k-1))*np.exp(-x/float(th))
    return wt_fcn(k, th)+1

def randomGaussianWeights(xy, a=None, b=None):
    """
    xy is nw-by-2
    mu and cov parameterize the 2d gaussian

    returns a vector of weights, where the weight of a
    point is given by the pdf of a 2d gaussian at that location.
    """
    a = 10*np.random.random() - 5 if a is None else a
    b = 1 if b is None else b
    mu = np.mean(xy, 0) + a
    cov = np.std(xy, 0) * b
    return scipy.stats.multivariate_normal.pdf(xy, mu, cov)

if __name__ == '__main__':
    from stim import Stim
    from plot import plotFull
    s = Stim(100, 7, 25)
    r = Resp(s, signalType='rank-3')
    # plotFull(s.xy, r.wt[:,None].dot(r.ws[None,:]))
    plotFull(s.xy, r.wf)
