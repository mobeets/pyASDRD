import numpy as np
import sklearn.cross_validation
import sklearn.linear_model
from asdard import ASD_FP

class Fit(object):
    def __init__(self, X0, Y0, X1=None, Y1=None, label=None):
        """
        X0, Y0 is training data
        X1, Y1 is testing data (defaults to training data if not supplied)
        """
        self.X0 = X0
        self.X1 = X1 if X1 is not None else X0
        self.Y0 = Y0
        self.Y1 = Y1 if Y1 is not None else Y0
        self.label = label if label is not None else ''
        self.clf = self.init_clf()
        return self

    def init_clf(self):
        raise NotImplementedError("clf not implemented!")

    def fit(self):
        self.clf.fit(self.X0, self.Y0)
        return self

    def predict(self):
        self.Yh1 = self.clf.predict(self.X1)
        return self

    def score(self, verbose=True):
        self.rsq = self.clf.score(self.X1, self.Y1)
        if verbose:
            self.print_score()
        return self

    def print_score(self):
        print '{0} score = {1}'.format(self.label, self.rsq)

class OLS(Fit):
    def init_clf(self):
        return sklearn.linear_model.LinearRegression(fit_intercept=False, normalize=False)

class Ridge(Fit):
    """
    Ridge regression with priors on tuning parameters
    """
    def init_clf(self):
        # (clf.alpha_, clf.lambda_)
        return sklearn.linear_model.BayesianRidge()

class Lasso(Fit):
    """
    Lasso model fit with tuning set by cross-validation
    """
    def init_clf(self):
        # clf.alpha_
        return sklearn.linear_model.LassoCV(fit_intercept=False, normalize=False)

class ARD(Fit):
    def init_clf(self):
        # (clf.alpha_, clf.lambda_)
        return sklearn.linear_model.ARDRegression(fit_intercept=False, normalize=False)

class Bilinear(Fit):
    def init_clf(self):
        # (clf.coef1_, clf.coef2_)
        return BilinearClf()

class BilinearClf(object):
    def fit(self, X, Y, niters=1000):
        whs = OLS(np.sum(X, 1), Y).fit().clf.coef_
        for _ in xrange(niters):
            wht = OLS(X.dot(whs), Y).fit().clf.coef_
            whs = OLS(wht.dot(X), Y).fit().clf.coef_
        self.coef1_ = wht
        self.coef2_ = whs
        self.coef_ = np.outer(wht, whs)

    def predict(self, X1):
        return self.coef1_.dot(X1).dot(self.coef2_)

    def score(self, X1, Y1):
        resid = lambda a, b: ((a-b)**2).sum()
        return 1 - resid(Y1, self.predict(X1))/resid(Y1, Y1.mean())

class ASD(Fit):
    def __init__(self, *args, **kwargs):
        self.Ds = kwargs.pop('Ds')
        self.Dt = kwargs.pop('Dt', None)
        super(ASD, self).__init__(*args, **kwargs)

    def init_clf(self):
        # (clf.coef_, clf.hyper_, clf.Reg_)
        return ASDClf(self.Ds, self.Dt)

class ASDClf(object):
    def __init__(self, Ds, Dt=None):
        self.Ds = Ds
        self.Dt = Dt
        self.D = [self.Ds] if Dt is None else [self.Ds, self.Dt]

    def fit(self, X, Y, maxiters=10000, step=0.01, tol=1e-6):
        self.coef_, self.Reg_, self.hyper_ = ASD_FP(X, Y, self.D)

    def predict(self, X1):
        return X1.dot(self.coef_)

    def score(self, X1, Y1):
        resid = lambda a, b: ((a-b)**2).sum()
        return 1 - resid(Y1, self.predict(X1))/resid(Y1, Y1.mean())

def trainAndTest(X, Y, trainPct=0.9):
    """
    returns (X,Y) split into training and testing sets
    """
    X0, X1, Y0, Y1 = sklearn.cross_validation.train_test_split(X, Y, train_size=trainPct, random_state=17)
    return (X0, Y0), (X1, Y1)

def kFold(X, Y, K=10, shuffle=False):
    """
    X, Y are training set data
    K is int
    returns sets for k-fold cross-validation:
        1. divide X, Y into K sets, for each pair K_i, ~K_i
        2. find best lambda_i on ~K_i, calculate error on K_i
        3. for each solution lambda_i, calculate error on each K_i
        4. choose lambda_i with minimum value in 3.
    """
    kf = sklearn.cross_validation.KFold(len(Y), n_folds=K, shuffle=shuffle)
    for train_index, test_index in kf:
        X0, X1 = X[train_index], X[test_index]
        Y0, Y1 = Y[train_index], Y[test_index]
        yield (X0, Y0), (X1, Y1)

if __name__ == '__main__':
    X = np.arange(10,50)
    Y = np.arange(20,60)
    for t,v in kFoldCV(zip(X, Y), 10):
        print t, v
