import numpy as np
import scipy.optimize

def ASDReg(ro, D, delta_sq):
    return np.exp(-ro-0.5*D/(delta_sq**2))

def ASDPostCov(RegInv, XX, sigma_sq):
    return np.linalg.inv((XX/sigma_sq) + RegInv)

def ASDPostMean(PostCov, XY, sigma_sq):
    return PostCov.dot(XY)/sigma_sq

def ASDEviGradient((ro, delta_s, sigma_sq), X, Y, XX, XY, p, q, D):
    """
    gradient of log evidence w.r.t. hyperparameters
    """
    # update regularizer
    S = ASDReg(ro, D, delta_s)
    S_inv = np.linalg.inv(S)

    # posterior cov and mean
    sigma =  ASDPostCov(S_inv, XX, sigma_sq)
    mu = ASDPostMean(sigma, XY, sigma_sq)
    err = Y - X.dot(mu)
    sse = err.dot(err.T)
    
    # hyperparameter derivatives
    Z = (S - sigma - mu.dot(mu.T)) * S_inv
    v = -p + np.trace(np.eye(q) - sigma*S_inv)
    # Z is too big after first iteration because mu.dot(mu.T) overpowers...
    der_ro = 0.5*np.trace(Z)
    der_delta_s = -0.5*np.trace(Z * np.multiply(S, D/(delta_s**3)) * S_inv)
    der_sigma_sq = sse/(sigma_sq**2) + v/sigma_sq

    return (der_ro, der_delta_s, der_sigma_sq), (mu, sigma, S, S_inv, sse)

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

def ASD(X, Y, D):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    D - (q, q) matrix containing squared distances between input points

    Returns:
        w - (q x 1) the regularized solution
        S - (q x q) the ASD regularizer
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """

    def objfun_maker(X, Y, XX, XY, p, q, D):
        def objfcn((ro, delta_s, sigma_sq)):
            hyper = (ro, delta_s, sigma_sq)
            der_hyper, (mu, sigma, S, S_inv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, D)
            evi = ASDEvi(X, Y, S, sigma, sigma_sq, p, q)
            return evi, -np.array(der_hyper)
        return objfcn

    p, q = np.shape(X)
    XY = X.T.dot(Y)
    XX = X.T.dot(X)
    
    #initialize parameters
    ro = 8.0
    delta_s = 2.0
    sigma_sq = 0.1 # np.sum(np.power(Y - X*(np.linalg.qr(X)[1]*Y),2)) / p    
    theta0 = (ro, delta_s, sigma_sq)

    objfcn = objfun_maker(X, Y, XX, XY, p, q, D)
    theta = scipy.optimize.minimize(objfcn, theta0, method='L-BFGS-B', jac=True)
    print theta
    der_hyper, (mu, sigma, S, S_inv, sse) = ASDEviGradient(theta['x'], X, Y, XX, XY, p, q, D)

    return mu, S

def ASD_FP(X, Y, D):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    D - (q, q) matrix containing squared distances between input points

    Returns:
        w - (q x 1) the regularized solution
        S - (q x q) the ASD regularizer
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """
    p, q = np.shape(X)
    XY = X.T.dot(Y)
    XX = X.T.dot(X)
    
    #initialize parameters
    ro = 8
    delta_s = 2.0
    sigma_sq = 0.1 # np.sum(np.power(Y - X*(np.linalg.qr(X)[1]*Y),2)) / p   
    der_ro = 0
    der_ro_m = 0
    der_delta_s_m = 0
    der_sigma_sq_m = 0
    
    step = 0.01
    print 'i, ro, delta_s, sigma_sq, der_ro_m, der_delta_s_m, step'
    for i in xrange(100):
        print i, ro, delta_s, sigma_sq, der_ro_m, der_delta_s_m, step, der_ro
        
        hyper = (ro, delta_s, sigma_sq)
        der_hyper, (mu, sigma, S, S_inv, sse) = ASDEviGradient(hyper, X, Y, XX, XY, p, q, D)
        (der_ro, der_delta_s, der_sigma_sq) = der_hyper
        
        if der_ro_m*der_ro + der_delta_s_m*der_delta_s < 0:
            step *= 0.8
            der_ro_m = der_ro
            der_delta_s_m = der_delta_s
        else:
            memstep = lambda base, prev: ((prev*base) > 0) * prev * 0.99
            der_ro_m = der_ro + memstep(der_ro, der_ro_m)
            der_delta_s_m = der_delta_s + memstep(der_delta_s, der_delta_s_m)

        ro += step * der_ro_m
        delta_s += step * der_delta_s_m
        delta_s = np.max([0.5, np.abs(delta_s)])
        v = -p + np.trace(np.eye(q) - sigma*S_inv)
        sigma_sq = -sse/v

    hyper = (ro, delta_s, sigma_sq)
    der_hyper, (mu, sigma, S, S_inv, sse) = ASDHyperGradient(hyper, X, Y, XX, XY, p, q, D)
    return mu, S

def ARD(X, Y):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    
    Implelements the ARD regression, adapted from:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """
    (p,q) = np.shape(X)
    
    #initialize parameters
    sigma_sq = 0.1
    CC = X.T * X
    XY = X.T * Y
    start_flag = False
        
    alpha = np.mat(np.zeros((q,1)))+2.0
    
    for i in xrange(0,100):
        sigma = np.linalg.inv(CC/sigma_sq + np.diagflat(alpha)) 
        ni = sigma * (XY) /  (sigma_sq)
        sigma_sq = np.sum(np.power(Y - X*ni,2))/(p - np.sum(1 - np.multiply(np.mat(np.diagonal(sigma)).T,alpha)))
        print np.min(np.abs(ni))
        alpha =  np.mat(np.divide((1 - np.multiply(alpha,np.mat(np.diagonal(sigma)).T)) , np.power(ni,2)))
        print  sigma_sq
        
    w = np.linalg.inv(CC + sigma_sq * np.diagflat(alpha)) * (XY)
    return w
    
def ASDRD(X, Y, S):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    
    Implelements the ARD regression, adapted from:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """
    D,V = np.linalg.eigh(S)
    V = np.mat(V)
    D = np.diag(np.sqrt(D))
    R =  V*  D * V.T
    w = ARD(X*R,Y)
    w = R * w
    return w
