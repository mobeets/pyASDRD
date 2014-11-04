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

def load(infile, N, M):
    """
    N is int - number of trials
    M is int - number of lags
    """
    fmt = os.path.splitext(infile)[-1].lower()
    if fmt == '.csv':
        xs = np.loadtxt(open(infile, "rb"), delimiter=",", skiprows=0)
        # xs = pd.read_csv(infile, index_col=None)
    elif fmt == '.h5':
        xs = pd.read_hdf(infile, 'table').values # assumes table named df
    elif fmt == '.npy':
        xs = np.load(infile)
    X = xs[:N, -(M+2):-2]
    Y = xs[:N, -1]
    ONES = xs[:N, -2]
    return X, Y

def main(infile, N=200, M=50, doPlot=False):
    X, Y = load(infile, N, M)
    (X0, Y0), (X1, Y1) = reg.trainAndTest(X, Y, trainPct=0.8)

    nt = X0.shape[1]
    xy = np.array(zip(np.arange(nt), np.zeros(nt))) # distances between lags in time is just 1s
    D = stim.sqdist(xy) # distance matrix

    # ARD
    obj = reg.ARD(X0, Y0, X1, Y1, label='ARD').fit().score()
    wf = obj.clf.coef_
    print wf
    if doPlot:
        plt.plot(wf, 'o')
        plt.show()

    # ASD
    obj = asdard.ASD(X0, Y0, X1, Y1, Ds=D, label='ASD').fit().score()
    wf = obj.clf.coef_
    print wf
    if doPlot:
        plt.plot(wf, 'o')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--infile", type=str, default=None)
    parser.add_argument("--convert", action='store_true', default=False)
    parser.add_argument("--plot", action='store_true', default=False)
    parser.add_argument("--fmt", type=str, choices=['h5', 'npy'], default=None)
    parser.add_argument("-m", type=int, default=50, help="# of trials to use")
    parser.add_argument("-n", type=int, default=50, help="# of frame lags to use")
    args = parser.parse_args()
    if args.convert:
        write(args.infile, args.fmt)
    else:
        main(args.infile, args.n, args.m, args.plot)
