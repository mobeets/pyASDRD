import scipy.io

def load(infile):
    assert infile.endswith('.mat')
    d = scipy.io.loadmat(infile)
    return d['X'], d['Y'], d['Xxy']
