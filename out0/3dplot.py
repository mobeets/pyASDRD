import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def color_list(n, cmap=None, gray=True):
    if gray:
        colors = [str(i) for i in np.linspace(0, 1, n)]
    else:
        cm = plt.get_cmap("RdYlGn" if cmap is None else cmap)
        colors = [cm(i) for i in np.linspace(0, 1, n)]
    return colors*(n/len(colors)) + colors[:n%len(colors)]

def load():
    df = pd.read_csv('evidences.csv')
    return df.ix[:255]

def plot(df, xkey='0', ykey='1', zkey='3', grpkey='2'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    colors = color_list(len(df[grpkey].unique()))
    for i, (grp, dfc) in enumerate(df.groupby(grpkey)):
        print len(dfc)
        ax.scatter(dfc[xkey].values, dfc[ykey].values, dfc[zkey].values, c=colors[i])#, zdir='y')
    plt.xlabel('ro')
    plt.ylabel('ssq')
    ax.set_zlabel('log evidence')
    plt.show()

def main():
    df = load()
    plot(df)

if __name__ == '__main__':
    main()
