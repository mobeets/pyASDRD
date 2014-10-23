import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

class Resp:
    def __init__(self, S, ssq=12.0, wt=None, ws=None, wf=None, signalType='bilinear'):
        """
        S.X is space-time stimulus on each trial
        S.xy is x,y locations of space as represented in stimulus
        ssq is variance of noise in response
        wt, ws, wf are the time, space, or full weights, respectively
        signalType in ['bilinear', 'spacey', 'full', 'rank-k'] where k is int
        
        calculates the weighted response to the given stimulus
        """
        self.ssq = ssq
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
    wfs = np.zeros([nt, ns, k])
    lastFirstBigger = False
    isFirstBigger = lambda wt: wt[:(nt/2)].sum() > wt[-(nt/2):].sum()
    for i in xrange(k):
        wt, ws = randomBilinear(nt, xy, norm=True)
        if lastFirstBigger and isFirstBigger(wt):
            # reverse time weight to keep them distinct
            wt = wt[::(2*(i%2) - 1)]
        lastFirstBigger = isFirstBigger(wt)
        wf = np.vstack(wt).dot(np.vstack(ws).T)
        wfs[:,:,i] = wf/wf.max()
    return wfs.sum(2)

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

def plot(xy, ws, sz=0.5*1e2):
    ws = ws/ws.max() if ws.max() > 0.0 else np.array([0.0]*len(ws))
    ws = 1.0 - ws
    cs = [str(w) for w in ws] # [(w, w, w) for w in ws]
    plt.scatter(xy[:,0], xy[:,1], s=sz, c=cs, lw=0)
    tm = xy[xy[:,0] == xy[0,0], 1]
    dist = np.abs(tm.mean() - tm.min())/2.
    plt.xlim(xy[:,0].min() - dist, xy[:,0].max() + dist)
    plt.ylim(xy[:,1].min() - dist, xy[:,1].max() + dist)
    plt.gca().set_aspect('equal')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().tick_params(axis='y', labelleft=False, left=False, right=False)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('0.8')

def plotFull(xy, wf):
    nt = wf.shape[0]
    plt.figure(figsize=(2,8), facecolor="white")
    for i in xrange(nt):
        plt.subplot(nt, 1, i+1)
        plot(xy, wf[i,:])
        plt.ylabel('t={0}'.format(i), rotation='horizontal', horizontalalignment='right')
    plt.show()

if __name__ == '__main__':
    from stim import Stim
    s = Stim(100, 7, 25)
    r = Resp(s, signalType='rank-3')
    plotFull(s.xy, r.wf)
