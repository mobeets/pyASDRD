import sys
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import reg
import stim
import resp
import plot
import asdard

def load(infile, N, M):
    """
    N is int - number of trials
    M is int - number of lags
    """
    # mat = sio.loadmat(infile)
    df = pd.read_csv(infile, index_col=None)
    return df.ix[:N,:M].values, df.ix[:N,-1].values, None

def main(infile, N=1000, M=50):
    X, Y, xy = load(infile, N, M)
    (X0, Y0), (X1, Y1) = reg.trainAndTest(X, Y, trainPct=0.8)

    nt = X0.shape[1]
    xy = np.array(zip(np.arange(nt), np.zeros(nt))) # distances between lags in time is just 1s
    D = stim.sqdist(xy) # distance matrix

    # ARD
    obj = reg.ARD(X0, Y0, X1, Y1, label='ARD').fit().score()
    wf = obj.clf.coef_
    print wf
    # plt.plot(wf, 'o')
    # plt.show()

    # ASD
    obj = asdard.ASD(X0, Y0, X1, Y1, Ds=D, label='ASD').fit().score()

    wf = obj.clf.coef_
    # plt.plot(wf, 'o')
    # plt.show()

if __name__ == '__main__':
    infile = os.path.abspath(sys.argv[-1].strip())
    main(infile)
