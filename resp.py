import numpy as np
import scipy.stats

class Resp:
    def __init__(self, S, ssq=8.0, wt=None, ws=None, signalType='bilinear'):
        """
        S.X is space-time stimulus on each trial
        S.xy is x,y locations of space as represented in stimulus
        ssq is variance of noise in response
        wt, ws are the time and space weights, respectively
        signalType is either 'bilinear', 'spacey', or 'full' [default]
        
        returns the space-time separable weighted response to the given stimulus
        """
        self.ssq = ssq
        (n, nt, ns) = S.X.shape
        self.signalType = signalType
        self.wt = randomTimeWeights(nt) if wt is None else wt
        self.ws = 10*randomGaussianWeights(S.xy) if ws is None else ws
        self.wf = randomFullRank(nt, ns)
        self.signalType = signalType
        if self.signalType == 'bilinear':
            self.ws = self.ws*(np.sum(self.wt)/np.sum(self.ws)) # scale so space and time roughly equal
        self.sig_fcn_lkp = {'bilinear': self.bilinear_signal, 'spacey': self.spacey_signal, 'full': self.full_signal}
        self.sig_fcn = self.sig_fcn_lkp.get(self.signalType, self.full_signal)
        self.Y = self.resp(S.X, self.wf, self.wt, self.ws, self.ssq)

    def full_signal(self, X, wf=None, wt=None, ws=None):
        return np.einsum('abc,bc -> a', X, wf)

    def bilinear_signal(self, X, wf=None, wt=None, ws=None):
        return wt.dot(X).dot(ws)

    def spacey_signal(self, X, wf=None, wt=None, ws=None):
        return np.sum(X, 1).dot(ws)

    def resp(self, X, wf, wt, ws, ssq):
        (n, nt, ns) = X.shape
        self.Ysig = self.sig_fcn(X, wf, wt, ws) # signal
        self.Ynse = np.random.normal(0, np.sqrt(ssq), n) # noise
        return self.Ysig + self.Ynse

def predict(X, w1, w2=None):
    if w2 is None:
        if len(w1.shape) == 1 or w1.shape[1] == 1:
            if X.shape[-1] == w1.shape[0]:
                return X.dot(w1)
            elif X.shape[0] == w1.shape[0]:
                return w1.dot(X)
        elif (X.shape[-1] == w1.shape[-1]) and (X.shape[-2] == w1.shape[-2]):
            return np.einsum('abc,bc -> a', X, w1)
    elif X.shape[-1] == w2.shape[0]:
        return w1.dot(X).dot(w2)
    elif X.shape[0] == w2.shape[0]:
        return w2.dot(X).dot(w1)
    raise Exception("Bad shapes = no prediction.")

def randomFullRank(nt, ns):
    return np.random.rand(nt, ns)

def randomTimeWeights(nt, k=5, th=1):
    x = np.arange(nt)
    wt_fcn = lambda k, th: (x**(k-1))*np.exp(-x/float(th))
    return wt_fcn(k, th)

def randomGaussianWeights(pts, mu=None, cov=None):
    """
    returns a vector of weights, where the weight of a
    point is given by the pdf of a 2d gaussian at that location.
    
    pts is nw-by-2; if false, pts will be randomly chosen
    mu and cov parameterize the 2d gaussian
    """
    if mu is None:
        mu = np.mean(pts, 0) + 1
    if cov is None:
        cov = np.std(pts, 0)/3.0
    return scipy.stats.multivariate_normal.pdf(pts, mu, cov)
