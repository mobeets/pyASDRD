import numpy as np
import matplotlib.pyplot as plt

def plot(S, R, i=0):
    if i == 0:
        plt.plot(np.sum(S.Xb, 1), R.Y, '.')
        # plt.plot(np.sum(S.Xb, 1), '.')
        # plt.plot(R.Y, '.')
        plt.show()

    elif i == 1:
        plt.plot(np.sum(S.Xb, 1), '.')
        plt.plot(R.Y, '.')
        plt.show()

    elif i == 2:
        plt.imshow(np.var(S.X, 0))
        plt.set_cmap('gray')
        plt.colorbar()
        plt.show()

def T(pcts, rs, ls, cs):
    for r,l,c in zip(rs, ls, cs):
        plt.plot(pcts, r, '-' + c, label=l)
    plt.xlabel('%')
    plt.ylabel('rmse')
    plt.legend()
    # plt.xscale('log')
    # plt.yscale('log')
    plt.show()

def XY(S, R):
    phi = np.linspace(0.0,2*np.pi,100)
    na = np.newaxis
    x = S.xy[:,0]
    y = S.xy[:,1]
    r = R.ws**0.5
    x_line = x[na,:] + r[na,:]*np.sin(phi[:,na])
    y_line = y[na,:] + r[na,:]*np.cos(phi[:,na])
    plt.plot(x_line, y_line, '-k')

    plt.plot(x, y, '.k', alpha=0.5)
    plt.gca().axis('equal')
    lim = (S.xy.min(), S.xy.max())
    plt.xlim(lim)
    plt.ylim(lim)
    plt.tight_layout()
    plt.show()

def plotFullInner(xy, ws, vmax=None, sz=0.5*1e2):
    vmax = ws.max() if vmax is None else vmax
    ws = ws/vmax if vmax > 0.0 else np.array([0.0]*len(ws))
    cs = [str(w) for w in 1.0-ws] # [(w, w, w) for w in ws]
    plt.scatter(xy[:,0], xy[:,1], s=sz, c=cs, lw=0)
    # pad
    tm = xy[xy[:,0] == xy[0,0], 1]
    dist = np.abs(tm.mean() - tm.min())/2.
    plt.xlim(xy[:,0].min() - dist, xy[:,0].max() + dist)
    plt.ylim(xy[:,1].min() - dist, xy[:,1].max() + dist)
    # format axes
    plt.gca().set_aspect('equal')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().tick_params(axis='y', labelleft=False, left=False, right=False)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('0.8')

def plotFull(xy, wf):
    nt = wf.shape[0]
    plt.figure(figsize=(8,2), facecolor="white")
    for i in xrange(nt):
        plt.subplot(1, nt, i+1)
        plotFullInner(xy, wf[i,:], vmax=wf.max())
        plt.title('t={0}'.format(i))#, rotation='horizontal', horizontalalignment='right')
    plt.show()
