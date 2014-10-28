import numpy as np
import scipy.optimize

def confine_to_bounds(theta, bounds):
    ts = []
    for i, ((lb, ub), tc) in enumerate(zip(bounds, theta)):
        tc = lb if (tc < lb and lb is not None) else tc
        tc = ub if (tc > ub and ub is not None) else tc
        ts.append(tc)
    return tuple(ts)

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

    # update regularizer
    Reg = ASDReg(ro, ds)
    RegInv = np.linalg.inv(Reg)
    if np.isinf(Reg).all():
        raise Exception("Reg is inf.")
    if np.isnan(RegInv).all():
        raise Exception("RegInv is nan.")

    # posterior cov and mean
    sigma =  PostCov(RegInv, XX, sigma_sq)
    mu = PostMean(sigma, XY, sigma_sq)
    err = Y - X.dot(mu)
    sse = err.dot(err.T)
    
    # hyperparameter derivatives
    Z = (Reg - sigma - np.outer(mu, mu)).dot(RegInv)
    v = -p + np.trace(np.eye(q) - sigma.dot(RegInv))
    der_ro = 0.5*np.trace(Z)
    der_sigma_sq = sse/(sigma_sq**2) + v/sigma_sq
    der_deltas = []
    for (D, d) in ds:
        der_deltas.append(-0.5*np.trace(Z.dot(Reg * D/(d**3)).dot(RegInv)))
    der_hyper = (der_ro, der_sigma_sq) + tuple(der_deltas)

    if np.isnan(np.array(der_hyper)).any():
        raise Exception("der_hyper is nan: {0}".format(der_hyper))
    return der_hyper, (mu, sigma, Reg, RegInv, sse)

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

def ASD(X, Y, Ds, theta0=None, method='SLSQP'):
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
            der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
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
    der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
    return mu, Reg, hyper

def ASD_FP(X, Y, Ds, theta0=None, maxiters=10000, step=0.01, tol=1e-6):
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
    theta_bounds = [(-20.0, 20.0), (None, None)] + [(0.5, None)]*len(Ds)

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

        der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
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
        deltas = np.abs(deltas)
        deltas[deltas < 0.5] = 0.5
        v = -p + np.trace(np.eye(q) - sigma*RegInv)
        sigma_sq = -sse/v

        hyper_prev = hyper
        hyper = (ro, sigma_sq) + tuple(deltas)
        hyper = confine_to_bounds(hyper, theta_bounds)
        if (np.abs(np.array(hyper_prev) - np.array(hyper)) < tol).all():
            break

    der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, Ds)
    return mu, Reg, hyper

def ARD(X, Y, niters=10000, tol=1e-6, alpha_thresh=1e-4):
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
        RegInv = np.diag(alpha)
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
    
def ASDRD(X, Y, RegASD):
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
    R = V.dot(np.diag(np.sqrt(D))).dot(V.T) # R = RegASD^(1/2)
    wp, RegInvP, _ = ARD(X.dot(R), Y)
    w = R.dot(wp)
    RegInv = R.dot(RegInvP).dot(R.T)
    return w, RegInv
