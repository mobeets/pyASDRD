import numpy as np
from tools import linv, rinv, PostCov

getS = lambda ro, delta_s, D: np.exp(-ro-0.5*D/(delta_s**2))

def hyperstep(der_ro, der_ro_m, der_delta_s, der_delta_s_m, step):
    # print der_ro, der_ro_m, der_delta_s, der_delta_s_m, step
    memstep = lambda base, prev: (prev*base > 0) * prev * 0.99
    if der_ro_m * der_ro + der_delta_s_m * der_delta_s < 0:
        step *= 0.8
        der_ro_m = der_ro
        der_delta_s_m = der_delta_s
    else:
        der_ro_m = der_ro + memstep(der_ro, der_ro_m)
        der_delta_s_m = der_delta_s + memstep(der_delta_s, der_delta_s_m)
    return step, der_ro_m, der_delta_s_m

def ASD_OG_OG(X, Y, D, theta0=None, maxiters=1000, step=0.01, tol=1e-5):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    D - (q,q) matrix containing Dances between input points
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """
    ro, ssq, delta_s = theta0 if theta0 is not None else (8.0, 0.1, 2.0)
    (p,q) = np.shape(X)    
    XX = X.T * X
    XY = X.T * Y
    
    der_ro_m = 0
    der_delta_s_m = 0
    hyper = np.array([ro, ssq, delta_s])
    for i in xrange(maxiters):
        print i, hyper

        S = getS(ro, delta_s, D)
        sigma =  np.linalg.inv(XX/ssq + np.linalg.inv(S))
        ni = sigma*XY/ssq
        sse = np.power(Y - X*ni, 2).sum()
        ssq = sse/(p - np.trace(np.eye(q) - rinv(S, sigma)))

        Z = rinv(S, S - sigma - ni*ni.T)
        der_ro = np.trace(Z)/2.0
        der_delta_s = -np.trace(rinv(S, Z * np.multiply(S, D/(delta_s**3))))/2.0
        step, der_ro_m, der_delta_s_m = hyperstep(der_ro, der_ro_m, der_delta_s, der_delta_s_m, step)
        ro += step * der_ro_m
        delta_s += step * der_delta_s_m
        delta_s = np.abs(delta_s)

        hyper_prev = hyper
        hyper = np.array([ro, ssq, delta_s])
        if (np.abs(hyper_prev - hyper) < tol).all():
            break

    S = getS(ro, delta_s, D)
    w = np.linalg.inv(XX/ssq + np.linalg.inv(S))*XY/ssq
    print hyper
    return w, S, hyper
