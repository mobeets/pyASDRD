import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import reg
import asdard
from resp import Resp
from stim import Stim
from plot import plot, plotT

rmse = lambda x, y: np.sqrt(np.mean((y-x)**2))
def main():
    n, ns, nt = 1000, 19, 7
    S = Stim(n, nt, ns)
    R = Resp(S)
    # plot(S, R)
    
    rows = []
    pcts = np.linspace(10, 80, 8)
    ns = np.round((n*pcts)/100.)
    ssqs = np.linspace(1, 10, 3)
    ws = R.ws
    for j, si in enumerate(ssqs):
        R = Resp(S, si, R.wt, R.ws)
        for i, ni in enumerate(ns):
            print '{0} of {1}: {2}, {3}'.format(j*len(ns)+ i+1, len(ns)*len(ssqs), ni, si)

            X = S.Xs[1:ni, :]
            Y = R.Y[1:ni]
            D = S.D

            whs1 = reg.OLS(X, Y)
            whs2, _, _ = asdard.ARD(X, Y)
            whs3, RegASD, _ = asdard.ASD_FP(X, Y, D)
            whs4, _ = asdard.ASDRD(X, Y, RegASD)

            rows.append(('ols', si, ni, rmse(whs1, ws)))
            rows.append(('ard', si, ni, rmse(whs2, ws)))
            rows.append(('asd', si, ni, rmse(whs3, ws)))
            rows.append(('asdrd', si, ni, rmse(whs4, ws)))

            print 'here'

    df = pd.DataFrame(rows, columns=['name', 'ssq', 'n', 'rmse'])
    ylim = df.rmse.min(), df.rmse.max()
    for ssq, dfc in df.groupby('ssq'):
        plt.clf()
        ax = plt.gca()
        for name, dfcp in dfc.groupby('name'):
            dfcp.plot('n', 'rmse', ax=ax, label=name)
        plt.xlabel('n')
        plt.ylabel('rmse')
        plt.ylim(ylim)
        plt.title('ssq = {0}'.format(ssq))
        plt.legend()
        plt.savefig('/Users/mobeets/Desktop/ssq-{0}.png'.format(ssq))
        # plt.show()
    # 1/0

if __name__ == '__main__':
    main()
