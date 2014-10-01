import numpy as np

from resp import Resp
from stim import Stim
import linreg
from plot import plot, plotT

rmse = lambda x, y: np.sqrt(np.mean((y-x)**2))
def main():
    n, ns, nt = 1000, 19, 7
    S = Stim(n, nt, ns)
    R = Resp(S)
    plot(S, R)
    
    rs = []
    ss = []
    ts = []
    pcts = np.linspace(1, 100, 25)
    ns = np.round((n*pcts)/100.)
    ws = R.ws
    for ni in ns:

        X = S.Xs[1:ni, :]
        Y = R.Y[1:ni]
        D = S.D

        whs = linreg.solve(X, Y)
        # Rhr, whr = ridge(X, Y)
        # RhASD, whASD, theta = ASD(X, Y, D)

        rs.append(rmse(whs, ws))
        # ss.append(rmse(whr, ws))
        # ts.append(rmse(whASD, ws))
    plotT(pcts, rs)

if __name__ == '__main__':
    main()
