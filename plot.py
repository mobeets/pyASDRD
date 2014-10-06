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
