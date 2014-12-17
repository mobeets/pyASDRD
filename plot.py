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

def color_lkp(clrMid=None, clrNeg=None, clrPos=None):
    clrMid = np.array([0.95, 0.95, 0.95]) if clrMid is None else clrMid # middle color
    clrNeg = np.array([0.9, 0.3, 0.3]) if clrNeg is None else clrNeg # negative color
    clrPos = np.array([0.3, 0.3, 0.9]) if clrPos is None else clrPos # positive color
    clrf = lambda i, c0, c1: c0 + (i*1.0)*(c1-c0)
    clr = lambda i: clrf(i, clrMid, clrPos) if i >= 0.0 else clrf(-i, clrMid, clrNeg)
    return lambda xs: np.array([clr(x) for x in xs])

def plotFullFormat(ax=None):
    ax = plt.gca() if ax is None else ax
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis='y', labelleft=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_edgecolor('0.8')

def plotFull(xy, wf, vmax=None, sz=2e2):
    nt = wf.shape[0]
    vmax = abs(wf).max() if vmax is None else vmax
    plt.figure(figsize=(16,4), facecolor='white')
    clrlkp = color_lkp()
    for i in xrange(nt):
        ax = plt.subplot(1, nt, i+1)
        # plotFullInnerNorm(xy, wf[i,:], vmax=vmax, clrlkp=clrlkp, ax=ax)
        plt.scatter(xy[:,0], xy[:,1], c=clrlkp(wf[i,:]/vmax), s=sz, lw=0)
        plotFullFormat(ax)
        plt.title('t={0}'.format(i))#, rotation='horizontal', horizontalalignment='right')
    plt.show()
