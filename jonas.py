import os.path
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pandas.tools.plotting import autocorrelation_plot, lag_plot

import reg
import stim
import resp
import plot
from asdard import ASDHyperGrid, ASDQuickPost, ASDLogLikelihood

"""
* grid up hyperparameters, assess MAP (PostCov, PostMu), calculate:
    - 1) evidence at that point; 2) likelihood on withheld data
"""

"""
* L1 regularization (e.g., lasso), L2 reg (e.g., ridge)
* L1-norm regularization => "sparse"
* L1-norm regularization better for logistic regression when there are more irrelevant features than training examples
    - see Ng, 2004
"""

def write(infile, fmt):
    assert infile.endswith('.csv')
    outfile = '{0}.{1}'.format(os.path.splitext(infile)[0], fmt)
    print 'Creating "{0}".'.format(outfile)
    if fmt == 'h5':
        df = pd.read_csv(infile, index_col=None)
        df.to_hdf(outfile, 'table', append=True)
    elif fmt == 'npy':
        xs = np.loadtxt(open(infile, "rb"), delimiter=",", skiprows=0)
        np.save(outfile, xs)

def load(infile, keep_ones=True):
    fmt = os.path.splitext(infile)[-1].lower()
    if fmt == '.csv':
        xs = np.loadtxt(open(infile, "rb"), delimiter=",", skiprows=0)
        # xs = pd.read_csv(infile, index_col=None)
    elif fmt == '.h5':
        xs = pd.read_hdf(infile, 'table').values # assumes table named df
    elif fmt == '.npy':
        xs = np.load(infile)
    end_ind = -1 if keep_ones else -2
    X = xs[:,:end_ind][:,::-1] # all but last columns; reverse columns
    Y = xs[:,-1] # last column
    return X, Y

def plot_lags(X, Y, lag=False, autocorr=False):
    X = X[:1000,:]
    Y = Y[:1000]
    X1 = X[:,0]
    X2 = X[:,200]
    # X2 = X[:,-5]
    # X3 = X[:-1,-10]
    # xs = np.diff(xs[:,:-2])
    # for i in xrange(1,4):
    #     plt.plot(xs[:,-i], label=str(i))
    plt.plot(X1, 'c', lw=1, alpha=0.5)
    plt.plot(X2, 'b', lw=1, alpha=0.5)
    # plt.plot(X3, 'g', lw=1, alpha=0.5)
    plt.plot(Y, 'r', lw=1, alpha=0.5)
    # plt.plot((Y-X)[:200], 'r', alpha=0.5)
    # plt.plot(X, Y, 'o')
    plt.show()
    return
    ind = -3
    if lag:
        lag_plot(pd.Series(X))
        plt.show()
    if autocorr:
        autocorrelation_plot(pd.Series(X))
        plt.show()

def split(X, Y, N, M, skipM=1, startM=0, front=True):
    """
    N is int - number of trials
    M is int - number of lags
    """
    if front:
        if skipM:
            X = X[:N, startM:startM + M*skipM:skipM]
        else:
            X = X[:N, startM:startM+M]
        Y = Y[:N]
    else:
        if skipM:
            X = X[-(N+1):, startM:startM + M*skipM:skipM]
        else:
            X = X[-(N+1):, startM:startM+M]
        Y = Y[-(N+1):]
    return X, Y

def main_kFold(infile, M=250, K=10, thresh=0.5):
    X, Y = load(infile, keep_ones=False)
    if thresh is not None:
        inds = np.abs(np.diff(np.hstack([0.0, Y]))) > thresh
        X = X[inds,:]
        Y = Y[inds]
    dX = np.diff(X)
    M = X.shape[1] if M is None else M
    items = []
    for i in xrange(M):
        if i % 10 == 0:
            print i
        Xc = X[:,i][:,None]
        for j, (X0, Y0, X1, Y1) in enumerate(reg.kFold(Xc, Y, K=K, shuffle=False)):
            obj = reg.OLS(X0, Y0, X1, Y1, label='OLS', fit_intercept=True).fit().score(verbose=False)
            items.append((i, j, obj.rsq) + (obj.clf.intercept_,) + tuple(obj.clf.coef_))
    df = pd.DataFrame(items, columns=['i', 'j', 'rsq', 'intercept', 'w1'])
    df.to_csv('/Users/mobeets/Desktop/lag_data_2.csv')

def temporal_distance(nt):
    xy = np.array(zip(np.arange(nt), np.zeros(nt))) # distances between lags in time is just 1s
    return stim.sqdist(xy) # distance matrix

def scores(X0, Y0, X1, Y1, D, theta, label):
    evi, mu, Reg, _ = ASDQuickPost(X0, Y0, [D], theta)
    obj = reg.ASD(X0, Y0, X1, Y1, Ds=D, fit_intercept=False)
    obj.clf.manual_fit(X0, Y0, mu, Reg, theta)
    score = obj.predict().score(verbose=False).rsq
    ll = ASDLogLikelihood(Y1, X1, mu, theta[1])
    print '{0}: evi={1}, rsq={2}, ll={3}'.format(label, evi, score, ll)
    return evi, ll, score

def loadit(infile, p, N, M, skipM, flip):
    X, Y = load(infile, keep_ones=False)
    if p > 0.0:
        N = int(p*X.shape[0])
    if p == 0.0 and N == 0:
        N = X.shape[0]
    if M == 0:
        M = X.shape[1]
    X0, Y0 = split(X, Y, N, M, skipM, front=flip)
    X1, Y1 = split(X, Y, X.shape[0] - N, M, skipM, front=not flip)
    D = temporal_distance(X0.shape[1])
    print 'Loaded.'
    return X0, Y0, X1, Y1, D

def evidence_on_hyper_grid(infile, p=0.1, N=0, M=0, skipM=0, flip=False):
    X0, Y0, X1, Y1, D = loadit(infile, p, N, M, skipM, flip)
    # hyper0 = (8.552115024753979, 1.7377065001127789, 1.0) # ASD
    # hyper0 = (3.552115024753979, 4.2377065001127789, 1.0)
    # hyper0 = (3.552115024753979, 6.7377065001127789, 1.0)
    # hyper0 = (3.552115024753979, 9.2377065001127789, 1.0)
    # hyper0 = (3.552115024753979, 11.737706500112779, 1.0)
    # hyper0 = (6.052115024753979, 4.2377065001127789, 1.0)
    # hyper0 = (6.052115024753979, 6.7377065001127789, 1.0)
    # thetaManual = (6.052115024753979, 6.7377065001127789, 1.0)
    # thetaManual = (10.21161596, 2.2431597, 2.40294741)
    thetaManual = (8.52268124, 1.67921395e+01, 1.00100000e-05)

    if True:
        # evi, ll, score = scores(X0, Y0, X1, Y1, D, thetaManual, 'Manual')
        obj = reg.ASD(X0, Y0, X1, Y1, Ds=D, label='ASD_OG', fit_intercept=False).fit(theta0=thetaManual, useFP=False)
        # obj = reg.ASD(X0, Y0, X1, Y1, Ds=D, label='ASD_OG_OG', fit_intercept=False).fit(theta0=thetaManual, useFP='OG_OG')
        theta = obj.clf.hyper_
        evi, ll, score = scores(X0, Y0, X1, Y1, D, theta, 'ASD_OG from manual')
    
    if False:
        # obj = reg.Ridge(X0, Y0, label='Ridge', fit_intercept=True).fit()
        # theta = np.array([-np.log(obj.clf.lambda_), 1./obj.clf.alpha_, 1.0])
        # evi, ll, score = scores(X0, Y0, X1, Y1, D, theta, 'Ridge')

        theta = thetaManual
        evi, ll, score = scores(X0, Y0, X1, Y1, D, theta, 'Manual')

        # obj = reg.ASD(X0, Y0, X1, Y1, Ds=D, label='ASD', fit_intercept=False).fit(theta0=theta)
        # theta = obj.clf.hyper_
        # evi, ll, score = scores(X0, Y0, X1, Y1, D, theta, 'ASD from manual')

        obj = reg.ASD(X0, Y0, X1, Y1, Ds=D, label='ASD_OG_OG', fit_intercept=False).fit(theta0=theta, useFP='OGOG')
        theta = obj.clf.hyper_
        evi, ll, score = scores(X0, Y0, X1, Y1, D, theta, 'ASD_OG_OG from manual')

    if False:
        theta1 = list(thetaManual)
        theta2 = list(thetaManual)
        theta3 = list(thetaManual)
        theta2[-1] = 5.0
        theta3[-1] = 10.0
        _, wf1, _, _ = ASDQuickPost(X0, Y0, [D], theta1)
        _, wf2, _, _ = ASDQuickPost(X0, Y0, [D], theta2)
        _, wf3, _, _ = ASDQuickPost(X0, Y0, [D], theta3)
        plt.plot(wf1, 'o', label='d={0}'.format(theta1[-1]), lw=2)
        plt.plot(wf2, 'o', label='d={0}'.format(theta2[-1]), lw=2)
        plt.plot(wf3, 'o', label='d={0}'.format(theta3[-1]), lw=2)
        plt.show()
        
    if False:
        hyper0 = thetaManual
        ds = [2.0, 3.0, 0.99]
        evis = ASDHyperGrid(X0, Y0, [D], n=3, hyper0=hyper0, ds=ds)
        eviM = max(evis, key=lambda x: x[-1])
        print 'Max grid evidence: {0}'.format(eviM)

def next_zoom(hyper0, ds0, hyper1, n):
    grid = [np.linspace(x-d, x+d, n) for x, d in zip(hyper0, ds0)]
    inds = [np.where(xs==x)[0][0] for xs, x in zip(grid, hyper1)]
    ds1 = [(xs[ind] - xs[ind-1]) if ind-1 >= 0 else (xs[ind+1] - xs[ind]) for ind, xs in zip(inds, grid)]
    return np.abs(np.array(ds1))

def max_evidence_zoom_grid(infile, p=0.1, N=0, M=0, skipM=0, flip=False):
    """
    TO DO:
        * resolve asd_fp and asd_og_og - shouldn't they be exact?
        * asd_og_og does not use fixed point update of ssq?
        * why come asd_og_og doesn't need SVD trick? maybe asd_fp doesn't either
        * figure out why asd_og gradient (jac) makes it not work

    OG, SLSQP, NO JAC:
        [  9.26162768  32.41763939   0.76989484] -2966.81535078
        [ 10.74723836  29.71702145   1.79645431] -2954.61520109
        [ 11.05848739  20.85859283   2.42059303] -2914.23949698
    OG, L-BFGS-B, NO JAC SOLUTION (95 iterations, success):
        [  8.52268124e+00   1.67921395e+01   1.00100000e-05] -2882.23518453
            evi=-2882.23518453, rsq=0.887728268089, ll=-5928.66544223
    NEW FIXED ASD_FP (56 iters, no SVD needed)
        (10.21161596, 2.2431597, 2.40294741)
            evi=-4988.06792921, rsq=0.886087419067, ll=-45013.8487565

    GRID PICK
        (14.158681239999998, 14.089472833333334, 4.0456699999999994)
            evi=-1200.59451625
        (13.244014573333331, 14.089472833333334, 3.36653)
            evi=-875.689985004
        (15.073347906666664, 12.260139500000001, 4.0456699999999994)
            evi=3705.26923941
        (15.073347906666664, 12.260139500000001, 3.8192899999999996)
            evi=110688.712398

    (with gradient, L-BFGS-B gives 'ABNORMAL_TERMINATION_IN_LNSRCH' error)

    START:
    [6.052115024753979, 6.7377065001127789, 2.0]
    [2.0, 3.0, 0.99]

    -2992.0682442619172
    [ 8.05211502  9.7377065   1.01      ]
    [ 1.4    2.1    0.693]

    -2914.8319250783693)
    [  9.45211502  11.8377065    1.241     ]
    [ 0.98    1.47    0.4851]

    -2895.3566399937172)
    [  9.77878169  13.3077065    1.4027    ]
    [ 0.686    1.029    0.33957]

    -2887.4399289761382)
    [ 10.00744836  14.3367065    1.74227   ]
    [ 0.4802    0.7203    0.237699]
    """
    # import itertools
    X0, Y0, _, _, D = loadit(infile, p, N, M, skipM, flip)

    nhypers = 3
    nbins = 4
    nzooms = 4
    deltas = np.zeros([nzooms, nhypers])
    centers = np.zeros([nzooms+1, nhypers])
    evidences = np.zeros([nzooms*(nbins**nhypers), nhypers+1])
    deltas[0,:] = np.array([4.0, 4.0, 0.99])
    # centers[0,:] = np.array([6.052115024753979, 6.7377065001127789, 2.0])
    centers[0,:] = np.array([8.52268124, 1.67921395e+01, 2.0])

    for i in xrange(nzooms):
        delta = deltas[i,:]
        center = centers[i,:]
        print delta
        print center
        print '-----'

        # hs = itertools.product(*[np.linspace(x-d, x+d, nbins) for x, d in zip(center, delta)])
        # evis = [(h, np.random.rand()) for h in hs]
        evis = ASDHyperGrid(X0, Y0, [D], n=nbins, hyper0=center, ds=delta)
        eviM = max(evis, key=lambda x: x[-1])
        print 'Max grid evidence: {0}'.format(eviM)

        for j, (h, e) in enumerate(evis):
            evidences[i*(nbins**nhypers)+j,:] = np.hstack([np.array(h), np.array([e])])

        nextCenter = eviM[0]
        centers[i+1,:] = nextCenter
        if i+1 < nzooms:
            deltas[i+1,:] = next_zoom(center, delta, nextCenter, nbins)*1.05 # perturb slightly
            deltas[i+1,-1] = np.min([deltas[i+1,-1], nextCenter[-1] + 1e-5]) # keep above 0
        print '======'
    pd.DataFrame(deltas).to_csv('out/deltas.csv')
    pd.DataFrame(centers).to_csv('out/centers.csv')
    pd.DataFrame(evidences).to_csv('out/evidences.csv')
    # 1/0

fitfcns = {'ols': reg.OLS, 'ridge': reg.Ridge, 'ard': reg.ARD, 'asd': reg.ASD, 'asdrd': reg.ASDRD, 'lasso': reg.Lasso}
def main(infile, fits, p=0.8, N=200, M=50, skipM=0, thresh=None, doPlot=False, label=None, flip=False, color=None, fitIntercept=False):
    """
    to initialize for ASD:
        * ssq = ridge.clf.alpha
        * ro = -log(ridge.clf.lambda)
        * d = 1
    """
    X, Y = load(infile, keep_ones=False)
    # 1/0

    useXDeriv = False
    ind = 41
    if useXDeriv:
        X0 = X[:,ind:ind+1]
        dX = np.diff(X)
        X = np.hstack([X0, dX])
    elif False:
        X0 = X[:,ind+1] - X[:,ind]
        X1 = X[:,ind+2] - X[:,ind+1]
        X2 = X1 - X0
        X3 = X[:,ind:ind+1]
        X = np.hstack([X1[:,None], X3])
        # X = np.hstack([X0[:,None], X1[:,None], X2[:,None], X])
        # X = np.hstack([X0[:,None], X1[:,None], X])
        # X = np.hstack([X0[:,None], X])

    print X.shape
    if thresh is not None:
        inds = np.abs(np.diff(np.hstack([0.0, Y]))) > thresh
        X = X[inds,:]
        Y = Y[inds]
        print X.shape

    if p > 0.0:
        N = int(p*X.shape[0])
    if p == 0.0 and N == 0:
        N = X.shape[0]
    if M == 0:
        M = X.shape[1]
    X0, Y0 = split(X, Y, N, M, skipM, front=flip)
    X1, Y1 = split(X, Y, X.shape[0] - N, M, skipM, front=not flip)
    print X0.shape

    if 'asd' in fits or 'asdrd' in fits:
        D = temporal_distance(X0.shape[1])

    wfs = {}
    asdreg = None
    theta0 = None
    for fit in sorted(fits):
        lbl = fit.upper()
        print 'Fitting {0}'.format(fit.upper())
        if fit == 'asd':
            theta0 = np.array([15.073347906666664, 12.260139500000001, 3.8192899999999996])
            print theta0
            evi, mu, Reg, _ = ASDQuickPost(X0, Y0, [D], theta0)
            print evi
            obj = reg.ASD(X0, Y0, X1, Y1, Ds=D, fit_intercept=False)
            obj.clf.manual_fit(X0, Y0, mu, Reg, theta0)
            
            # rdg = fitfcns['ridge'](X0, Y0, X1, Y1, label=lbl, fit_intercept=fitIntercept).fit()
            # theta0 = np.array([-np.log(rdg.clf.lambda_), 1./rdg.clf.alpha_, 1.0])
            
            # obj = fitfcns[fit](X0, Y0, X1, Y1, Ds=D, label=lbl, fit_intercept=fitIntercept).fit(theta0=theta0).score()
            asdreg = obj.clf.Reg_
        elif fit == 'asdrd':
            obj = fitfcns[fit](X0, Y0, X1, Y1, Ds=D, asdreg=asdreg, label=lbl, fit_intercept=fitIntercept).fit(theta0=theta0).score()
        else:
            obj = fitfcns[fit](X0, Y0, X1, Y1, label=lbl, fit_intercept=fitIntercept).fit().score()
        print obj.clf.intercept_
        wfs[fit] = obj.clf.coef_
    print obj.clf.coef_.sum()
    print obj.clf.coef_[1:].sum()

    if doPlot:
        for fit, wf in wfs.iteritems():
            lbl = fit if label is None else label
            lw = 3
            alpha = 0.7
            if skipM:
                ms = np.arange(len(wf))*skipM
                plt.plot(ms, wf, '-', label=lbl, lw=lw, alpha=alpha)
            else:
                plt.plot(wf, '-', label=lbl, lw=lw, alpha=alpha)

if __name__ == '__main__':
    ALL_FITS = fitfcns.keys()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--infile", type=str, default=None)
    parser.add_argument("--convert", action='store_true', default=False)
    parser.add_argument("--plot", action='store_true', default=False)
    parser.add_argument("--intercept", action='store_true', default=False, help="Fit intercept")
    parser.add_argument("--fmt", type=str, choices=['h5', 'npy'], default=None)
    parser.add_argument("--fits", default=ALL_FITS, nargs='*', choices=ALL_FITS, type=str, help="The fitting methods you would like to use, from: {0}".format(ALL_FITS))
    parser.add_argument("-m", type=int, default=50, help="# of frame lags to use")
    parser.add_argument("--skipm", type=int, default=0, help="# of lags to skip")
    parser.add_argument("-n", type=int, default=200, help="# of trials to use")
    parser.add_argument("-p", type=float, default=0.1, help="%% of trials to use for training data")
    parser.add_argument("--thresh", type=float, default=None, help="eye position diff threshold")
    args = parser.parse_args()
    if args.convert:
        write(args.infile, args.fmt)
    elif False:
        max_evidence_zoom_grid(args.infile)
    elif False:
        evidence_on_hyper_grid(args.infile)
    else:
        cmap = matplotlib.cm.get_cmap('Reds')
        main(args.infile, args.fits, args.p, args.n, args.m, args.skipm, args.thresh, args.plot, fitIntercept=args.intercept)
        if args.plot:
            plt.plot(plt.xlim(), [0, 0], '--', color='gray')
            plt.xlabel('foe lag')
            plt.ylabel('weight')
            # plt.ylim(-0.02, 0.04)
            plt.legend(loc='upper right')
            plt.gcf().patch.set_facecolor('white')
            plt.show(block=False)
            raw_input("Hit Enter To Close ")
            plt.close()
