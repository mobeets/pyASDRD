
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
