import numpy as np
import scipy.optimize
from asdard import ASDEvi, ASDPosterior, ASDEviGradient, asd_theta_bounds

def ASD_OG(X, Y, Ds, theta0=None, jac=False, method='L-BFGS-B'): # 'SLSQP'
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
    theta0 = (-1.0, 0.1) + (2.0,)*len(Ds) if theta0 is None else theta0    
    bounds = [(-20.0, 20.0), (10e-6, 10e6), (0, 10e3)]
    # bounds = asd_theta_bounds(Ds)
    p, q = X.shape
    XY = X.T.dot(Y)
    XX = X.T.dot(X)

    def objfcn(hyper, jac=jac):
        mu, sigma, Reg, sse = ASDPosterior(hyper, X, Y, XX, XY, Ds)
        evi = ASDEvi(X, Y, Reg, sigma, hyper[1], p, q)
        if not jac:
            return -evi
        der_hyper = ASDEviGradient(hyper, p, q, Ds, mu, sigma, Reg, sse)
        print hyper, der_hyper, evi
        return -evi, -np.array(der_hyper)

    theta = scipy.optimize.minimize(objfcn, theta0, bounds=bounds, method=method, jac=jac)
    if not theta['success']:
        print theta
    hyper = theta['x']
    print hyper
    mu, _, Reg, _ = ASDPosterior(hyper, X, Y, XX, XY, Ds)
    return mu, Reg, hyper
