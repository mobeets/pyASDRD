import numpy as np
import scipy.optimize


def PostCov(RegInv, XX, sigma_sq):
    return np.linalg.inv((XX/sigma_sq) + RegInv)

def PostMean(sigma, XY, sigma_sq):
    return sigma.dot(XY)/sigma_sq

def ASDReg(ro, D, delta_s):
    return np.exp(-ro-0.5*D/(delta_s**2))

def ASDEviGradient((ro, delta_s, sigma_sq), X, Y, XX, XY, p, q, D):
    """
    gradient of log evidence w.r.t. hyperparameters
    """
    # update regularizer
    Reg = ASDReg(ro, D, delta_s)
    RegInv = np.linalg.inv(Reg)

    # posterior cov and mean
    sigma =  PostCov(RegInv, XX, sigma_sq)
    mu = PostMean(sigma, XY, sigma_sq)
    err = Y - X.dot(mu)
    sse = err.dot(err.T)
    
    # hyperparameter derivatives
    Z = (Reg - sigma - np.outer(mu, mu)).dot(RegInv)
    v = -p + np.trace(np.eye(q) - sigma.dot(RegInv))
    # Z is too big after first iteration because mu.dot(mu.T) overpowers...
    der_ro = 0.5*np.trace(Z)
    # print Reg.shape, D.shape
    # print np.trace(Z), np.diag(Reg * D/(delta_s**3)), np.diag(D/(delta_s**3)), np.diag(Reg)
    der_delta_s = -0.5*np.trace(Z.dot(Reg * D/(delta_s**3)).dot(RegInv))
    der_sigma_sq = sse/(sigma_sq**2) + v/sigma_sq

    return (der_ro, der_delta_s, der_sigma_sq), (mu, sigma, Reg, RegInv, sse)

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

def ASD(X, Y, D, theta0=(8.0, 2.0, 0.1), method='SLSQP'):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    D - (q, q) matrix containing squared distances between input points

    Returns:
        w - (q x 1) the regularized solution
        Reg - (q x q) the ASD regularizer
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """

    def objfun_maker(X, Y, XX, XY, p, q, D):
        def objfcn((ro, delta_s, sigma_sq)):
            hyper = (ro, delta_s, sigma_sq)
            der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, D)
            evi = ASDEvi(X, Y, Reg, sigma, sigma_sq, p, q)
            # print -evi, hyper, der_hyper
            return -evi, -np.array(der_hyper)
        return objfcn

    p, q = np.shape(X)
    XY = X.T.dot(Y)
    XX = X.T.dot(X)

    objfcn = objfun_maker(X, Y, XX, XY, p, q, D)
    # bounds = [(-20.0, 20.0), (10e-6, 10e6), (10e-6, 10e6)]
    bounds = [(-20.0, 20.0), (0.5, None), (1e-5, None)]
    theta = scipy.optimize.minimize(objfcn, theta0, bounds=bounds, method=method, jac=True)
    if not theta['success']:
        print theta
    hyper = theta['x']
    der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, D)
    return mu, Reg, hyper

def ASD_FP(X, Y, D, theta0=(8.0, 2.0, 0.1), maxiters=10000, step=0.01, tol=1e-6,):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    D - (q, q) matrix containing squared distances between input points

    Returns:
        w - (q x 1) the regularized solution
        Reg - (q x q) the ASD regularizer
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """
    memstep = lambda base, prev: ((prev*base) > 0) * prev * 0.99

    p, q = np.shape(X)
    XY = X.T.dot(Y)
    XX = X.T.dot(X)
    
    hyper = theta0
    der_ro = 0
    der_ro_m = 0
    der_delta_s_m = 0
    der_sigma_sq_m = 0
    for i in xrange(maxiters):
        (ro, delta_s, sigma_sq) = hyper
        der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, D)
        (der_ro, der_delta_s, der_sigma_sq) = der_hyper
        
        if der_ro_m*der_ro + der_delta_s_m*der_delta_s < 0:
            step *= 0.8
            der_ro_m = der_ro
            der_delta_s_m = der_delta_s
        else:
            der_ro_m = der_ro + memstep(der_ro, der_ro_m)
            der_delta_s_m = der_delta_s + memstep(der_delta_s, der_delta_s_m)

        ro += step * der_ro_m
        delta_s += step * der_delta_s_m
        delta_s = np.max([0.5, np.abs(delta_s)])
        v = -p + np.trace(np.eye(q) - sigma*RegInv)
        sigma_sq = -sse/v

        hyper_prev = hyper
        hyper = (ro, delta_s, sigma_sq)
        if (np.abs(np.array(hyper_prev) - np.array(hyper)) < tol).all():
            break
        # print hyper, der_hyper

    der_hyper, (mu, sigma, Reg, RegInv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, D)
    return mu, Reg, hyper

def ARD(X, Y, niters=10000, tol=1e-6):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    
    Implelements the ARD regression, adapted from:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """
    (p,q) = np.shape(X)
    XX = X.T.dot(X)
    XY = X.T.dot(Y)
    
    # initialize parameters
    sigma_sq = 0.1        
    alpha = 2*np.ones(q)

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
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """
    D, V = np.linalg.eigh(RegASD) # RegASD = V * D * V^T
    # V = np.mat(V)
    D2 = np.diag(np.sqrt(D)) # is this the correct interpretation of D2?
    R = V.dot(D2).dot(V.T)
    wp, RegInvP, _ = ARD(X.dot(R), Y)
    w = R.dot(wp)
    RegInv = R.dot(RegInvP).dot(R.T)
    return w, RegInv
