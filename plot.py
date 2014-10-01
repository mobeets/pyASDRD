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

def plotT(pcts, res):
    plt.plot(pcts, res, 'o')
    plt.xlabel('%')
    plt.ylabel('rmse')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.show()
