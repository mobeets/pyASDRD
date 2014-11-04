import os.path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import reg
import stim
import resp
import plot
import asdard

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

def load(infile,):
    fmt = os.path.splitext(infile)[-1].lower()
    if fmt == '.csv':
        xs = np.loadtxt(open(infile, "rb"), delimiter=",", skiprows=0)
        # xs = pd.read_csv(infile, index_col=None)
    elif fmt == '.h5':
        xs = pd.read_hdf(infile, 'table').values # assumes table named df
    elif fmt == '.npy':
        xs = np.load(infile)
    return xs

def split(xs, N, M, front=True):
    """
    N is int - number of trials
    M is int - number of lags
    """
    if front:
        X = xs[:N, -(M+2):-2]
        Y = xs[:N, -1]
        # ONES = xs[:N, -2]
    else:
        X = xs[-(N+1):, -(M+2):-2]
        Y = xs[-(N+1):, -1]
        # ONES = xs[-(N+1):, -2]
    return X, Y

fitfcns = {'ridge': reg.Ridge, 'ard': reg.ARD}
def main(infile, N=200, M=50, doPlot=False):
    xs = load(infile)
    # X, Y = split(xs, N, M)
    # (X0, Y0), (X1, Y1) = reg.trainAndTest(X, Y, trainPct=0.8)
    
    X0, Y0 = split(xs, N, M, front=True)
    X1, Y1 = split(xs, N, M, front=False)

    nt = X0.shape[1]
    xy = np.array(zip(np.arange(nt), np.zeros(nt))) # distances between lags in time is just 1s
    D = stim.sqdist(xy) # distance matrix

    wfs = {}
    fits = ['ridge', 'ard']
    for fit in fits:
        if fit == 'asd':
            obj = fitfcns[fit](X0, Y0, X1, Y1, Ds=D, label=fit.upper()).fit().score()
        else:
            obj = fitfcns[fit](X0, Y0, X1, Y1, label=fit.upper()).fit().score()
        wfs[fit] = obj.clf.coef_

    if doPlot:
        for fit, wf in wfs.iteritems():
            plt.plot(wf, 'o', label=fit)
        plt.plot(plt.xlim(), [0, 0], '--', color='gray')
        plt.legend(loc='lower left')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--infile", type=str, default=None)
    parser.add_argument("--convert", action='store_true', default=False)
    parser.add_argument("--plot", action='store_true', default=False)
    parser.add_argument("--fmt", type=str, choices=['h5', 'npy'], default=None)
    parser.add_argument("-m", type=int, default=50, help="# of trials to use")
    parser.add_argument("-n", type=int, default=200, help="# of frame lags to use")
    args = parser.parse_args()
    if args.convert:
        write(args.infile, args.fmt)
    else:
        main(args.infile, args.n, args.m, args.plot)
