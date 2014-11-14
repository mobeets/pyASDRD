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
import asdard

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
    return xs[:,:end_ind][:,::-1], xs[:,-1]

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

def split(X, Y, N, M, skipM=5, front=True):
    """
    N is int - number of trials
    M is int - number of lags
    """
    if front:
        if skipM:
            X = X[:N, :M*skipM:skipM]
        else:
            X = X[:N, :M]
        Y = Y[:N]
    else:
        if skipM:
            X = X[-(N+1):, :M*skipM:skipM]
        else:
            X = X[-(N+1):, :M]
        Y = Y[-(N+1):]
    return X, Y

fitfcns = {'ols': reg.OLS, 'ridge': reg.Ridge, 'ard': reg.ARD, 'asd': asdard.ASD, 'asdrd': asdard.ASDRD, 'lasso': reg.Lasso}
def main(infile, fits, p=0.8, N=200, M=50, skipM=0, thresh=None, doPlot=False, label=None, flip=False, color=None):
    """
    to initialize for ASD:
        * ssq = ridge.clf.alpha
        * ro = -log(ridge.clf.lambda)
        * d = 1
    """
    X, Y = load(infile, keep_ones=False)
    print X.shape
    if thresh is not None:
        inds = np.abs(np.diff(np.hstack([0.0, Y]))) > thresh
        X = X[inds,:]
        Y = Y[inds]
        print X.shape
    # plot_lags(X, Y)
    # return
    # 1/0

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
        nt = X0.shape[1]
        xy = np.array(zip(np.arange(nt), np.zeros(nt))) # distances between lags in time is just 1s
        D = stim.sqdist(xy) # distance matrix

    wfs = {}
    asdreg = None
    theta0 = None
    for fit in sorted(fits):
        print 'Fitting {0}'.format(fit.upper())
        if fit == 'asd':
            rdg = fitfcns['ridge'](X0, Y0, X1, Y1, label=fit.upper()).fit()
            theta0 = np.array([-np.log(rdg.clf.lambda_), 1./rdg.clf.alpha_, 1.0])
            print theta0
            obj = fitfcns[fit](X0, Y0, X1, Y1, Ds=D, label=fit.upper()).fit(theta0=theta0).score()
            asdreg = obj.clf.Reg_
        elif fit == 'asdrd':
            # assert asdreg is not None
            # assert theta0 is not None
            obj = fitfcns[fit](X0, Y0, X1, Y1, Ds=D, asdreg=asdreg, label=fit.upper()).fit(theta0=theta0).score()
        else:
            obj = fitfcns[fit](X0, Y0, X1, Y1, label=fit.upper()).fit().score()
        wfs[fit] = obj.clf.coef_

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
    parser.add_argument('-j', "--infile2", type=str, default=None)
    parser.add_argument("--convert", action='store_true', default=False)
    parser.add_argument("--plot", action='store_true', default=False)
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
    else:
        cmap = matplotlib.cm.get_cmap('Reds')
        main(args.infile, args.fits, args.p, args.n, args.m, args.skipm, args.thresh, args.plot)
        if args.infile2:
            cmap = matplotlib.cm.get_cmap('Blues')
            main(args.infile2, args.fits, args.p, args.n, args.m, args.skipm, args.thresh, args.plot, 'startY', color=cmap(0.7))

        if args.plot:
            plt.plot(plt.xlim(), [0, 0], '--', color='gray')
            plt.xlabel('foe lag')
            plt.ylabel('weight')
            # plt.ylim(-0.02, 0.04)
            plt.legend(loc='upper right')
            plt.gcf().patch.set_facecolor('white')
            plt.show()
