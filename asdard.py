import numpy as np
import scipy.optimize
import sklearn.metrics
from reg import Fit, ARD
from tools import rinv, woodbury

def confine_to_bounds(theta, bounds):
    ts = []
    for i, ((lb, ub), tc) in enumerate(zip(bounds, theta)):
        tc = lb if (tc < lb and lb is not None) else tc
        tc = ub if (tc > ub and ub is not None) else tc
        ts.append(tc)
    return tuple(ts)

def PostCov2(Reg, XT, sigma_sq):
    return woodbury(Reg, XT, sigma_sq)

def PostCov(RegInv, XX, sigma_sq):
    return np.linalg.inv((XX/sigma_sq) + RegInv)

def PostMean(sigma, XY, sigma_sq):
    return sigma.dot(XY)/sigma_sq

def ASDReg(ro, ds):
    """
    ro - float
    ds - list of tuples [(D, d), ...]
        D - (q x q) squared distance matrix in some stimulus dimension
        d - float, the weighting of D
    """
    vs = 0.0
    for D, d in ds:
        if not hasattr(vs, 'shape'):
            vs = np.zeros(D.shape)
        vs += D/(d**2)
    return np.exp(-ro-0.5*vs)

def PriorCovSVD(Reg, condThresh=1e10, eigenPctThresh=1-1e-5):
    """
    1. if Reg's condition is <= condThresh, return Reg
    2. if PCR on Reg results in a lower condition, return that PC-reduced Reg
    3. otherwise, return Reg
    """
    initCond = np.linalg.cond(Reg)
    if initCond < condThresh:
        return Reg, None
    U, s, Vh = scipy.linalg.svd(Reg, full_matrices=False)
    inds = np.cumsum(s)/s.sum() <= eigenPctThresh
    # inds = s/s.sum() > 1e-18

    RegNew = None
    while RegNew is None or (~np.isnan(np.linalg.cond(RegNew)) and np.linalg.cond(RegNew) > np.linalg.cond(Reg)):
        inds[np.argmax(~inds)] = True # add in one more eigenvalue
        B = U[:,inds]
        RegNew = B.T.dot(Reg).dot(B)
    if ~np.isnan(np.linalg.cond(RegNew)) and np.linalg.cond(RegNew) < np.linalg.cond(Reg):
        return RegNew, B
    return Reg, None

def ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds):
    """
    gradient of log evidence w.r.t. hyperparameters
    """
    ro, sigma_sq = hyper[:2]
    if np.isinf(np.exp(-ro)):
        raise Exception("Invalid ro: {0}".format(ro))
    deltas = hyper[2:]
    assert len(Ds) == len(deltas)
    ds = zip(Ds, deltas)

    # update regularizer; if regularizer is close to singular, use svd
    Reg = ASDReg(ro, ds)
    if np.isinf(Reg).all():
        raise Exception("Reg is inf.")
    Reg, B = PriorCovSVD(Reg)
    # map to new basis
    if B is not None:
        X = X.dot(B)
        XY = X.T.dot(Y)
        XX = X.T.dot(X)

    # posterior cov and mean
    sigma = PostCov(np.linalg.inv(Reg), XX, sigma_sq)
    mu = PostMean(sigma, XY, sigma_sq)

    # map back to original basis, if any
    if B is not None:
        X = X.dot(B.T)
        Reg = B.dot(Reg).dot(B.T)
        mu = B.dot(mu)
        sigma = B.dot(sigma).dot(B.T)
    
    # hyperparameter derivatives
    Z = rinv(Reg, Reg - sigma - np.outer(mu, mu))
    der_ro = 0.5*np.trace(Z)
    err = Y - X.dot(mu)
    sse = err.dot(err.T)
    v = -p + np.trace(np.eye(q) - rinv(Reg, sigma))
    der_sigma_sq = sse/(sigma_sq**2) + v/sigma_sq
    der_deltas = []
    for (D, d) in ds:
        der_deltas.append(-0.5*np.trace(rinv(Reg, Z.dot(Reg * D/(d**3)))))
    der_hyper = (der_ro, der_sigma_sq) + tuple(der_deltas)
    assert not np.isnan(np.array(der_hyper)).any(), "der_hyper is nan: {0}".format(der_hyper)
    return der_hyper, (mu, sigma, Reg, sse)

logDet = lambda x: np.linalg.slogdet(x)[1]
def ASDEvi(X, Y, Reg, sigma, sigma_sq, p, q):
    """
    log evidence
    """
    z1 = 2*np.pi*sigma
    z2 = 2*np.pi*sigma_sq*np.eye(q)
    z3 = 2*np.pi*Reg
    logZ = logDet(z1) - logDet(z2) - logDet(z3)
    B = (np.eye(p)/sigma_sq) - (X.dot(sigma).dot(X.T))/(sigma_sq**2)
    return 0.5*(logZ - Y.T.dot(B).dot(Y))

def ASD_OG(X, Y, Ds, theta0=None, method='SLSQP'):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    Ds - [(q, q), ...] matrices containing squared distances between input points

    Returns:
        w - (q x 1) the regularized solution
        Reg - (q x q) the ASD regularizer
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds.
        Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003
    """
    if theta0 is None:
        theta0 = (-1.0, 0.1) + (2.0,)*len(Ds)

    def objfun_maker(X, Y, XX, XY, p, q, Ds):
        def objfcn(hyper):
            # der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
            der_hyper, (mu, sigma, Reg, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
            sigma_sq = hyper[1]
            evi = ASDEvi(X, Y, Reg, sigma, sigma_sq, p, q)
            # print -evi, hyper, der_hyper
            return -evi, -np.array(der_hyper)
        return objfcn

    p, q = X.shape
    XY = X.T.dot(Y)
    XX = X.T.dot(X)

    objfcn = objfun_maker(X, Y, XX, XY, p, q, Ds)
    # bounds = [(-20.0, 20.0), (10e-6, 10e6)] + [(10e-6, 10e6)]*len(Ds)
    bounds = [(-20.0, 20.0), (1e-5, None)] + [(0.5, None)]*len(Ds)
    theta = scipy.optimize.minimize(objfcn, theta0, bounds=bounds, method=method, jac=True)
    if not theta['success']:
        print theta
    hyper = theta['x']
    # der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
    der_hyper, (mu, sigma, Reg, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
    return mu, Reg, hyper

def ASD_FP(X, Y, Ds, theta0=None, maxiters=10000, step=0.01, tol=1e-5):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    Ds - [(q, q), ...] matrices containing squared distances between input points

    Returns:
        w - (q x 1) the regularized solution
        Reg - (q x q) the ASD regularizer
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds.
        Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003
    """
    if theta0 is None:
        theta0 = (-1.0, 0.1) + (2.0,)*len(Ds)
    theta_bounds = [(-20.0, 20.0), (None, None)] + [(10e-6, 10e3)]*len(Ds)
    # theta_bounds = [(-20.0, 20.0), (None, None)] + [(0.5, 10e6)]*len(Ds)

    p, q = X.shape
    XY = X.T.dot(Y)
    XX = X.T.dot(X)
    
    hyper = theta0
    der_ro = 0
    der_ro_m = 0
    der_deltas_m = np.zeros(len(Ds))
    der_delta_s_m = 0
    der_sigma_sq_m = 0

    memstep = lambda base, prev: ((prev*base) > 0) * prev * 0.99
    for i in xrange(maxiters):
        ro, sigma_sq = hyper[:2]
        deltas = np.array(hyper[2:])

        der_hyper, (mu, sigma, Reg, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
        der_ro, der_sigma_sq = der_hyper[:2]
        der_deltas = np.array(der_hyper[2:])
        
        if der_ro_m*der_ro + (der_deltas_m*der_deltas).sum() < 0:
            step *= 0.8
            der_ro_m = der_ro
            der_deltas_m = der_deltas
        else:
            der_ro_m = der_ro + memstep(der_ro, der_ro_m)
            ddm2s = []
            for der_delta, der_delta_m in zip(der_deltas, der_deltas_m):
                ddm2s.append(der_delta + memstep(der_delta, der_delta_m))
            der_deltas_m = np.array(ddm2s)

        ro += step * der_ro_m
        deltas += step * der_deltas_m
        # deltas = np.abs(deltas)
        # deltas[deltas < 0.5] = 0.5
        v = -p + np.trace(np.eye(q) - rinv(Reg, sigma))
        sigma_sq = -sse/v

        hyper_prev = hyper
        hyper = (ro, sigma_sq) + tuple(deltas)
        hyper = confine_to_bounds(hyper, theta_bounds)
        if i % 5 == 0:
            # print '.',
            print
            print i, np.array(hyper) - np.array(hyper_prev)
            print i, np.array(hyper)
        if (np.abs(np.array(hyper_prev) - np.array(hyper)) < tol).all():
            break

    der_hyper, (mu, sigma, Reg, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
    print hyper
    return mu, Reg, hyper

def ARD2(X, Y, niters=10000, tol=1e-6, alpha_thresh=1e-4):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements

    n.b. should threshold alpha
    
    Implelements the ARD regression, adapted from:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds.
        Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003
    """
    (p,q) = X.shape
    XX = X.T.dot(X)
    XY = X.T.dot(Y)
    
    # initialize parameters
    sigma_sq = 0.1        
    alpha = 2*np.ones(q)

    keep_alpha = np.ones(q, dtype=bool)

    for i in xrange(niters):
        RegInv = np.diagflat(alpha)
        sigma = PostCov(RegInv, XX, sigma_sq)
        mu = PostMean(sigma, XY, sigma_sq)
        sse = np.power(Y - X.dot(mu), 2).sum()
        v = 1 - np.diagonal(sigma)*alpha

        old_alpha = alpha
        old_sigma_sq = sigma_sq
        alpha = v/(mu**2)
        sigma_sq = sse/(p - v.sum())
        if np.abs(sigma_sq - old_sigma_sq) < tol and np.abs(alpha - old_alpha).sum() < tol:
            break

    RegInv = np.diagflat(alpha)
    sigma = PostCov(RegInv, XX, sigma_sq)
    mu = PostMean(sigma, XY, sigma_sq)
    return mu, RegInv, (alpha, sigma_sq)
    
def ASDRD_inner(X, Y, RegASD):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    RegASD - (q x q) ASD regularizer solution
    
    Implelements ARD regression in an ASD basis (aka ASD/RD), adapted from:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds.
        Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003
    """
    D, V = np.linalg.eigh(RegASD) # RegASD = V * D * V^T
    if (np.abs(D[D<0]) < 1e-7).all():
        D[D<0] = 0
    else:
        raise ValueError("ASD reg has some large-ish negative eigenvalues")
    print "eigenvalue decomposition complete"
    R = V.dot(np.diag(np.sqrt(D))).dot(V.T) # R = V * sqrt(D) * V^T
    # wp, RegInvP, _ = ARD(X.dot(R), Y)
    print "next"
    # 1/0
    obj = ARD(X.dot(R), Y).fit()
    print "ARD complete"
    w = R.dot(obj.clf.coef_)
    print "next"
    RegInvP = np.diagflat(obj.clf.alpha_)
    print "finally..."
    # RegInvP = obj.clf.sigma_ # this may be wrong--might want np.diag(obj.clf.alpha_)
    # msk = obj.clf.lambda_ > obj.clf.threshold_lambda
    # R = R[~msk,:][:,~msk]
    RegInv = R.dot(RegInvP).dot(R.T)
    return w, RegInv

class ASD(Fit):
    def __init__(self, *args, **kwargs):
        self.Ds = kwargs.pop('Ds')
        self.Dt = kwargs.pop('Dt', None)
        super(ASD, self).__init__(*args, **kwargs)

    def init_clf(self):
        # (clf.coef_, clf.hyper_, clf.Reg_)
        return ASDClf(self.Ds, self.Dt, fit_intercept=self.fit_intercept)

class ASDClf(object):
    def __init__(self, Ds, Dt=None, fit_intercept=False):
        self.Ds = Ds
        self.Dt = Dt
        self.D = [self.Ds] if Dt is None else [self.Ds, self.Dt]
        self.fit_intercept = fit_intercept

    def center_data(self, X, Y):
        """
        source: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/base.py
        """
        if self.fit_intercept:
            X_mean = np.average(X, axis=0)
            Y_mean = np.average(Y, axis=0)
            X -= X_mean
            Y -= Y_mean
        else:
            X_mean = np.zeros(X.shape[1])
            Y_mean = 0. if Y.ndim == 1 else np.zeros(Y.shape[1], dtype=X.dtype)
        return X, Y, X_mean, Y_mean

    def set_intercept(self, X_mean, Y_mean):
        if self.fit_intercept:
            self.intercept_ = Y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.

    def fit(self, X, Y, theta0=None, maxiters=10000, step=0.01, tol=1e-6):
        X, Y, X_mean, Y_mean = self.center_data(X, Y)
        self.coef_, self.Reg_, self.hyper_ = ASD_FP(X, Y, self.D,
            theta0=theta0, maxiters=maxiters, step=step, tol=tol)
        self.set_intercept(X_mean, Y_mean)

    def predict(self, X1):
        return X1.dot(self.coef_) + self.intercept_

    def score(self, X1, Y1):
        return sklearn.metrics.r2_score(Y1, self.predict(X1))

class ASDRD(ASD):
    def __init__(self, *args, **kwargs):
        self.asdreg = kwargs.pop('asdreg', None)
        super(ASDRD, self).__init__(*args, **kwargs)

    def init_clf(self):
        return ASDRDClf(self.Ds, self.Dt, self.asdreg, fit_intercept=self.fit_intercept)

class ASDRDClf(ASDClf):
    def __init__(self, Ds, Dt=None, asdreg=None, fit_intercept=False):
        self.Ds = Ds
        self.Dt = Dt
        self.D = [self.Ds] if Dt is None else [self.Ds, self.Dt]
        self.asdreg = asdreg
        self.fit_intercept = fit_intercept

    def fit(self, X, Y, theta0=None, maxiters=10000, step=0.01, tol=1e-6):
        X, Y, X_mean, Y_mean = self.center_data(X, Y)
        if self.asdreg is None:
            self.asd_coef_, self.asdreg, self.asd_hyper_ = ASD_FP(X, Y, self.D,
            theta0=theta0, maxiters=maxiters, step=step, tol=tol)
            print "ASD complete"
        self.coef_, self.invReg_ = ASDRD_inner(X, Y, self.asdreg)
        self.set_intercept(X_mean, Y_mean)
