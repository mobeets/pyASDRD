import numpy as np
import scipy.optimize

# LOWER_BOUND_DELTA_TEMPORAL = 0.025904
LOWER_BOUND_DELTA_TEMPORAL = 0.12 # less than about 0.30 is indistinguishable

logDet = lambda x: np.linalg.slogdet(x)[1]
linv = lambda A, y: np.linalg.solve(A, y)
rinv = lambda A, y: np.linalg.solve(A.T, y.T).T

def PostCovInv(RegInv, XX, ssq):
    return XX/ssq + RegInv

def PostMean(SigmaInv, XY, ssq):
    return linv(SigmaInv, XY)/ssq

def ASDLogEviSVD(X, Y, YY, Reg, ssq, tol=1e-8):
    """
    calculate log-evidence in basis defined by eigenvalues of Reg > tol*S[0]
        where S[0] is largest eigenvalue
    i.e., if Reg is m x m and only k of Reg's eigenvalues meet this criteria,
        this function uses the rank-k approximation of Reg
    """
    U, s, _ = np.linalg.svd(Reg)
    inds = s/s[0] > tol
    # print 'SVD: using {0} cols of rank-{1} prior covariance matrix'.format(inds.sum(), len(inds))
    RegInv = np.diag(1/s[inds])
    B = U[:,inds]
    XB = X.dot(B)
    XBXB = XB.T.dot(XB)
    XBY = XB.T.dot(Y)
    SigmaInv = PostCovInv(RegInv, XBXB, ssq)
    return ASDLogEvi(XBXB, YY, XBY, np.diag(s[inds]), SigmaInv, ssq, XB.shape[0], XB.shape[1])

def ASDLogEvi(XX, YY, XY, Reg, SigmaInv, ssq, p, q):
    """
    XX is X.T.dot(X) - m x m
    YY is Y.T.dot(Y) - 1 x 1
    XY is X.T.dot(Y) - m x 1
    """
    A = -logDet(Reg.dot(XX)/ssq + np.eye(q)) - p*np.log(2*np.pi*ssq)
    B = YY/ssq - XY.T.dot(linv(SigmaInv, XY))/(ssq**2)
    return (A - B)/2.0

def ASDEviGradient(hyper, p, q, Ds, mu, Sigma, Reg, sse):
    """
    gradient of log evidence w.r.t. hyperparameters
    """
    ro, ssq = hyper[:2]
    deltas = hyper[2:]
    Z = rinv(Reg, Reg - Sigma - np.outer(mu, mu))
    der_ro = np.trace(Z)/2.0
    
    v = -p + q - np.trace(rinv(Reg, Sigma))
    der_ssq = sse/(ssq**2) + v/ssq

    der_deltas = []
    for (D, d) in zip(Ds, deltas):
        der_deltas.append(-np.trace(rinv(Reg, Z.dot(Reg * D/(d**3))))/2.0)
    return np.array((der_ro, der_ssq) + tuple(der_deltas))

def ASDLogLikelihood(Y, X, mu, ssq):
    sse = ((Y - X.dot(mu))**2).sum()
    return -sse/(2.0*ssq) - np.log(2*np.pi*ssq)/2.0

def RidgeReg(ro, q):
    return np.exp(-ro)*np.eye(q)

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

def ASDInit(X, Y, D, (ro, ssq, delta)):
    XX = X.T.dot(X)
    XY = X.T.dot(Y)
    YY = Y.T.dot(Y)
    p, q = X.shape
    Reg = ASDReg(ro, [(D, delta)])
    return XX, XY, YY, p, q, Reg

def MeanInvCov(XX, XY, Reg, ssq):
    SigmaInv = PostCovInv(np.linalg.inv(Reg), XX, ssq)
    return PostMean(SigmaInv, XY, ssq), SigmaInv

def evidence(X, Y, D, (ro, ssq, delta)):
    XX, XY, YY, p, q, Reg = ASDInit(X, Y, D, (ro, ssq, delta))
    SigmaInv = PostCovInv(np.linalg.inv(Reg), XX, ssq)
    return ASDLogEvi(XX, YY, XY, Reg, SigmaInv, ssq, p, q)

def loglikelihood(X, Y, D, (ro, ssq, delta)):
    XX, XY, YY, p, q, Reg = ASDInit(X, Y, D, (ro, ssq, delta))
    mu, _ = MeanInvCov(XX, XY, Reg, ssq)
    return ASDLogLikelihood(Y, X, mu, ssq)

def scores(X0, Y0, X1, Y1, D, (ro, ssq, delta)):
    evi = evidence(X0, Y0, D, (ro, ssq, delta))
    ll = loglikelihood(X1, Y1, D, (ro, ssq, delta))
    return evi, -ll

def ASD_inner(X, Y, Ds, theta0=None, verbose=False, jac=False, isLog=True, method='TNC'): # 'TNC' 'CG', 'SLSQP', 'L-BFGS-B'
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    Ds - [(q, q), ...] matrices containing squared distances between all input points
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds.
        Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003
    """
    theta0 = (1.0, 0.1) + (2.0,)*len(Ds) if theta0 is None else theta0
    if isLog:
        theta0 = np.log(theta0)
        bounds = [(-3.0, 3.0), (-2.0, 10.0)] + [(-5.0, 10.0)]*len(Ds)
    else:
        bounds = [(-20.0, 20.0), (10e-6, 10e6)] + [(LOWER_BOUND_DELTA_TEMPORAL, 1e5)]*len(Ds)
    p, q = X.shape
    YY = Y.T.dot(Y)
    XY = X.T.dot(Y)
    XX = X.T.dot(X)

    def objfcn(hyper, jac=jac, verbose=verbose):
        if isLog:
            hyper = np.exp(hyper)
        if verbose:
            print hyper
        ro, ssq = hyper[:2]
        deltas = hyper[2:]
        Reg = ASDReg(ro, zip(Ds, deltas))
        try:
            _ = np.linalg.cholesky(Reg) # raises exception if not positive definite
            SigmaInv = PostCovInv(np.linalg.inv(Reg), XX, ssq)
            logevi = ASDLogEvi(XX, YY, XY, Reg, SigmaInv, ssq, p, q)
        except np.linalg.LinAlgError:
            logevi = ASDLogEviSVD(X, Y, YY, Reg, ssq) # svd trick
        if not jac:
            return -logevi
        mu = PostMean(SigmaInv, XY, ssq)
        sse = (Y - X.dot(mu)**2).sum()
        der_logevi = ASDEviGradient(hyper, p, q, Ds, mu, np.linalg.inv(SigmaInv), Reg, sse)
        if verbose:
            print -np.array(der_logevi)
        return -logevi, -np.array(der_logevi)

    options = {'maxiter': int(1e8)}
    theta = scipy.optimize.minimize(objfcn, theta0, bounds=bounds, method=method, jac=jac, options=options)
    if not theta['success']:
        print theta
    hyper = theta['x']
    if isLog:
        hyper = np.exp(hyper)
    print hyper
    ro, ssq = hyper[:2]
    deltas = hyper[2:]
    Reg = ASDReg(ro, zip(Ds, deltas))
    mu, _ = MeanInvCov(XX, XY, Reg, ssq)
    return mu, Reg, hyper
