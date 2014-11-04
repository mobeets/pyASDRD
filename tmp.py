import reg
import stim
import resp
import asdard
import numpy as np

n = 120
ns, nt = 7, 25
SNR = 2.0

s = stim.Stim(n, ns, nt)
Ds = s.D
Ds = np.tile(Ds, [s.nt, s.nt])
Dt = stim.Dt(s.nt, s.ns)
r = resp.Resp(s, signalType='rank-2', SNR=SNR)
(X0, Y0), (X1, Y1) = reg.trainAndTest(s.Xf, r.Y, trainPct=0.8)

obj = asdard.ASD(X0, Y0, X1, Y1, Ds=Ds, Dt=Dt, label='{0} - SNR={1}'.format('ASD', SNR))
obj.fit().score(verbose=True)
obj = asdard.ASDRD(X0, Y0, X1, Y1, Ds=Ds, Dt=Dt, asdobj=obj, label='{0} - SNR={1}'.format('ASDRD', SNR))
obj.fit().score(verbose=True)
