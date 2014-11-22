import numpy as np
from tools import PostCov, PostMean

def ARD_OG(X, Y, niters=10000, tol=1e-6, alpha_thresh=1e-4):
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
