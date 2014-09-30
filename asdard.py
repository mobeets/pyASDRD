import numpy as np

def ASD(X, Y, dist):
    """
    X - (p x q) matrix with inputs in rows
    Y - (p, 1) matrix with measurements
    dist - (q, q) matrix containing distances between input points
    
    Implelements the ASD regression descrived in:
        M. Sahani and J. F. Linden.
        Evidence optimization techniques for estimating stimulus-response functions.
        In S. Becker, S. Thrun, and K. Obermayer, eds., Advances in Neural Information Processing Systems, vol. 15, pp. 301-308, Cambridge, MA, 2003. 
    """
    
    (p,q) = np.shape(X)
    
    #initialize parameters
    ro = 8
    delta_s = 2.0
    sigma_sq = 0.1 # np.sum(np.power(Y - X*(np.linalg.qr(X)[1]*Y),2)) / p
    print sigma_sq
    
    step = 0.01
    
    C = X.T * X
    XY = X.T * Y
    start_flag = False
    
    der_ro_m = 0
    der_delta_s_m = 0
    der_sigma_sq_m = 0
    
    for i in xrange(100):
        print i, ro, delta_s, sigma_sq
        
        S = np.exp(-ro-0.5*dist/(delta_s*delta_s))
        
        S_inv = np.linalg.inv(S)
        sigma =  np.linalg.inv(C /(sigma_sq) + S_inv)
        
        ni = sigma * (XY) /  (sigma_sq)
        
        Z = (S-sigma-(ni*ni.T)) * S_inv
        der_ro = np.trace(Z)
        der_delta_s = - np.trace(Z * np.multiply(S,dist/(np.power(delta_s,3))) * S_inv)
        
        if start_flag:
            der_ro_m = der_ro
            der_delta_s_m = der_delta_s
        else:
           if der_ro_m*der_ro + der_delta_s_m * der_delta_s < 0:
              step = step * 0.8
              der_ro_m = der_ro
              der_delta_s_m = der_delta_s
           else:
              
              der_ro_m = der_ro +  (der_ro_m * der_ro > 0) * der_ro_m * 0.99       
              der_delta_s_m = der_delta_s +  (der_delta_s_m * der_delta_s > 0) *der_delta_s_m * 0.99
        
        ro = ro + step * der_ro_m
        delta_s = delta_s + step * der_delta_s_m
    
        sigma_sq = np.sum(np.power(Y - X*ni,2))/(p - np.trace(np.eye(q) - sigma*S_inv));
        delta_s = np.max([0.5, np.abs(delta_s)])
    
    S = np.exp(-ro-0.5*dist/(delta_s*delta_s)) 
    S_inv = np.linalg.inv(S)
    w = np.linalg.inv(C + sigma_sq * S_inv) * (XY)
    
    return w, S

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
        sigma_sq = np.sum(np.power(Y - X*ni,2))/(p - np.sum(1 - np.multiply(np.mat(np.diagonal(sigma)).T,alpha)));
        print np.min(np.abs(ni))
        alpha =  np.mat(np.divide((1 - np.multiply(alpha,np.mat(np.diagonal(sigma)).T)) , np.power(ni,2)))
        print  sigma_sq
        
    w = np.linalg.inv(CC + sigma_sq * np.diagflat(alpha)) * (XY)
    
    print alpha
    print  sigma_sq
    
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
