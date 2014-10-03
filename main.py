import numpy as np

import linreg
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
    
    rs = []
    ss = []
    ts = []
    pcts = np.linspace(10, 80, 10)
    ns = np.round((n*pcts)/100.)
    ws = R.ws
    for i, ni in enumerate(ns):
        print '{0} of {1}: {2}'.format(i+1, len(ns), ni)

        X = S.Xs[1:ni, :]
        Y = R.Y[1:ni]
        D = S.D

        whs1 = linreg.solve(X, Y)
        whs2, _, _ = asdard.ARD(X, Y)
        whs3, _, _ = asdard.ASD_FP(X, Y, D)

        rs.append(rmse(whs1, ws))
        ss.append(rmse(whs2, ws))
        ts.append(rmse(whs3, ws))

        print 'here'

        # X2 = S.Xs[ni+1:, :]
        # Y2 = R.Y[ni+1:]

        # rs.append(rmse(X2.dot(whs1), Y2))
        # ss.append(rmse(X2.dot(whs2), Y2))
        # ts.append(rmse(X2.dot(whs3), Y2))

    plotT(pcts, [rs, ss, ts], ['ols', 'ard', 'asd'], ['b', 'g', 'r'])
    # 1/0

if __name__ == '__main__':
    main()
