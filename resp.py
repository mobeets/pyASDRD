import numpy as np
import scipy.stats
import scipy.spatial.distance

class Resp:
    def __init__(self, S, xy, ssq=3.0, wt=None, ws=None):
        """
        S is space-time stimulus on each trial
        pts is x,y locations of space as represented in stimulus
        ssq is variance of noise in response
        
        returns the space-time separable weighted response to the given stimulus
        """
        self.ssq = ssq
        self.D = self.dist(xy)
        (n, nt, ns) = S.shape
        self.wt = randomTimeWeights(nt) if wt is None else wt
        self.ws = 10*randomGaussianWeights(xy) if ws is None else ws
        # self.ws = self.ws*(sum(self.wt)/sum(self.ws)); % scale so space and time roughly equal
        self.R = self.resp(S, self.wt, self.ws, self.ssq)

    def dist(self, xy):
        return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(xy, 'euclidean'))

    def resp(self, S, wt, ws, ssq):
        (n, nt, ns) = S.shape
        sig = np.sum(S, 1).dot(ws) # signal
        nse = np.random.normal(0, np.sqrt(ssq), n) # noise
        return sig + nse

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
