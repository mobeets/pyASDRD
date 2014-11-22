import itertools
import scipy
import numpy as np
import sklearn.metrics
from tools import rinv, PostMean, PostCov

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
    if Reg's condition number is <= condThresh, return Reg
    if PCR on Reg results in a matrix with a lower condition number, return that PC-reduced Reg
    otherwise, return Reg
    """
    return Reg, None
    initCond = np.linalg.cond(Reg)
    if initCond < condThresh:
        return Reg, None
    U, s, Vh = scipy.linalg.svd(Reg, full_matrices=False)
    inds = np.cumsum(s)/s.sum() <= eigenPctThresh # s/s.sum() > 1e-18

    RegNew = None
    while RegNew is None or (~np.isnan(np.linalg.cond(RegNew)) and np.linalg.cond(RegNew) > np.linalg.cond(Reg)):
        inds[np.argmax(~inds)] = True # add in one more eigenvalue
        B = U[:,inds]
        RegNew = B.T.dot(Reg).dot(B)
    if ~np.isnan(np.linalg.cond(RegNew)) and np.linalg.cond(RegNew) < np.linalg.cond(Reg):
        return RegNew, B
    return Reg, None

def ASDRegSVD(ro, ds):
    """
    update regularizer; if regularizer is close to singular, use svd
    """
    Reg = ASDReg(ro, ds)
    if np.isinf(Reg).all():
        raise Exception("Reg is inf.")
    return PriorCovSVD(Reg)

def ASDPosteriorMeanAndCov(X, Y, XX, XY, Reg, B, ssq):
    # map to new basis, if any
    if B is not None:
        X = X.dot(B)
        XY = X.T.dot(Y)
        XX = X.T.dot(X)

    sigma = PostCov(np.linalg.inv(Reg), XX, ssq)
    mu = PostMean(sigma, XY, ssq)

    # map back to original basis, if any
    if B is not None:
        X = X.dot(B.T)
        Reg = B.dot(Reg).dot(B.T)
        mu = B.dot(mu)
        sigma = B.dot(sigma).dot(B.T)
    return mu, sigma

def unpackHypers(hyper, Ds):
    ro, ssq = hyper[:2]
    if np.isinf(np.exp(-ro)):
        raise Exception("Invalid ro: {0}".format(ro))
    deltas = hyper[2:]
    assert len(Ds) == len(deltas)
    ds = zip(Ds, deltas)
    return ro, ssq, ds

def ASDPosterior(hyper, X, Y, XX, XY, Ds):
    ro, ssq, ds = unpackHypers(hyper, Ds)
    Reg, B = ASDRegSVD(ro, ds)
    mu, sigma = ASDPosteriorMeanAndCov(X, Y, XX, XY, Reg, B, ssq)
    Reg = B.dot(Reg).dot(B.T) if B is not None else Reg
    err = Y - X.dot(mu)
    sse = err.dot(err.T)
    return mu, sigma, Reg, sse

def ASDEviGradient(hyper, p, q, Ds, mu, sigma, Reg, sse):
    """
    gradient of log evidence w.r.t. hyperparameters
    """
    ro, ssq, ds = unpackHypers(hyper, Ds)
    Z = rinv(Reg, Reg - sigma - np.outer(mu, mu))
    der_ro = np.trace(Z)/2.0
    
    # the below two lines are not even used!
    v = -p + np.trace(np.eye(q) - rinv(Reg, sigma))
    der_ssq = sse/(ssq**2) + v/ssq

    der_deltas = []
    for (D, d) in ds:
        der_deltas.append(-np.trace(rinv(Reg, Z.dot(Reg * D/(d**3))))/2.0)
    der_hyper = np.array((der_ro, der_ssq) + tuple(der_deltas))
    assert not np.isnan(der_hyper).any(), "der_hyper is nan: {0}".format(der_hyper)
    return der_hyper

def confine_to_bounds(theta, bounds):
    ts = []
    for i, ((lb, ub), tc) in enumerate(zip(bounds, theta)):
        tc = lb if (tc < lb and lb is not None) else tc
        tc = ub if (tc > ub and ub is not None) else tc
        ts.append(tc)
    return np.array(ts)

def hyperstep(der_ro, der_ro_m, der_deltas, der_deltas_m, step):
    # print der_ro, der_ro_m, der_deltas[0], der_deltas_m[0], step
    memstep = lambda base, prev: (prev*base > 0) * prev * 0.99
    if der_ro_m * der_ro + (der_deltas_m*der_deltas).sum() < 0:
        step *= 0.8
        der_ro_m = der_ro
        der_deltas_m = der_deltas
    else:
        der_ro_m = der_ro + memstep(der_ro, der_ro_m)
        der_deltas_m = np.array([d + memstep(d, dm) for d, dm in zip(der_deltas, der_deltas_m)])
    return step, der_ro_m, der_deltas_m

asd_theta_bounds = lambda Ds: [(-20.0, 20.0), (10e-6, 10e6)] + [(10e-6, 10e3)]*len(Ds)
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
    theta0 = theta0 if theta0 is not None else (8.0, 0.1) + (2.0,)*len(Ds)
    theta_bounds = asd_theta_bounds(Ds)
    p, q = X.shape
    XX = X.T.dot(X)
    XY = X.T.dot(Y)
    
    der_ro_m = 0
    der_deltas_m = np.zeros(len(Ds))

    hyper = np.array(theta0)
    for i in xrange(maxiters):
        print i, hyper
        ro, ssq, deltas = hyper[0], hyper[1], hyper[2:]

        # calculate posterior
        mu, sigma, Reg, sse = ASDPosterior(hyper, X, Y, XX, XY, Ds)
        ssq = sse/(p - np.trace(np.eye(q) - rinv(Reg, sigma)))

        # update hyperparameters
        der_hyper = ASDEviGradient(hyper, p, q, Ds, mu, sigma, Reg, sse)
        der_ro, der_deltas = der_hyper[0], der_hyper[2:]
        step, der_ro_m, der_deltas_m = hyperstep(der_ro, der_ro_m, der_deltas, der_deltas_m, step)
        ro += step * der_ro_m
        deltas += step * der_deltas_m
        deltas = np.abs(deltas)

        # check for bounds and convergence
        hyper_prev = hyper
        hyper = np.array([ro, ssq, deltas])
        hyper = confine_to_bounds(hyper, theta_bounds)
        if (np.abs(hyper_prev - hyper) < tol).all():
            break
    print hyper
    mu, _, Reg, _ = ASDPosterior(hyper, X, Y, XX, XY, Ds)
    return mu, Reg, hyper

def nextHyperInGridFromHyper(hyper, ds, n):
    rng = []
    for x, d in zip(hyper, ds):
        rng.append(np.linspace(x-d, x+d, n))
    for x in itertools.product(*rng):
        yield x

def nextHyperInGridFromBounds(bounds, n):
    """
    for each lower and upper bound in bounds,
    choose n equally spaced values within that range
    returns the cartesian product of all the resulting combos
    """
    rng = []
    for i, (lb, ub) in enumerate(bounds):
        rng.append(np.linspace(lb, ub, n))
    for x in itertools.product(*rng):
        yield x

def ASDLogLikelihood(Y, X, mu, ssq):
    sse = ((Y - X.dot(mu))**2).sum()
    return -sse/(2.0*ssq) - np.log(2*np.pi*ssq)/2.0

def ASDQuickPost(X, Y, Ds, hyper):
    mu, sigma, Reg, _ = ASDPosterior(hyper, X, Y, X.T.dot(X), X.T.dot(Y), Ds)
    evi = ASDEvi(X, Y, Reg, sigma, hyper[1], X.shape[0], X.shape[1])
    return evi, mu, sigma, Reg

def ASDHyperGrid(X, Y, Ds, n=5, hyper0=None, ds=None):
    """
    evaluates the evidence at each hyperparam
        within a grid of valid hyperparams
    """
    p, q = X.shape
    XX = X.T.dot(X)
    XY = X.T.dot(Y)
    evis = []
    if hyper0 is not None:
        grid = nextHyperInGridFromHyper(hyper0, ds, n)
    else:
        grid = nextHyperInGridFromBounds(asd_theta_bounds(Ds), n)
    for hyper in grid:
        print hyper
        ssq = hyper[1]
        _, sigma, Reg, _ = ASDPosterior(hyper, X, Y, XX, XY, Ds)
        evi = ASDEvi(X, Y, Reg, sigma, ssq, p, q)
        print evi
        print '-----'
        evis.append((hyper, evi))
    return evis

def ASDRD_inner(X, Y, RegASD, ARD):
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

if __name__ == '__main__':
    for hyper in nextHyperInGrid(asd_theta_bounds([1])):
        print hyper
