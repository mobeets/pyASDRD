import numpy as np
import sklearn.cross_validation
import sklearn.linear_model
from asdrd import ASDRD_inner
import asd
# from asd import ASD_inner

class Fit(object):
    def __init__(self, X0, Y0, X1=None, Y1=None, label=None, fit_intercept=False, normalize=False):
        """
        X0, Y0 is training data
        X1, Y1 is testing data (defaults to training data if not supplied)
        """
        self.X0 = X0
        self.X1 = X1 if X1 is not None else X0
        self.Y0 = Y0
        self.Y1 = Y1 if Y1 is not None else Y0
        self.label = label if label is not None else ''
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.clf = self.init_clf()
        # return self

    def init_clf(self):
        raise NotImplementedError("clf not implemented!")

    def fit(self, **kwargs):
        self.clf.fit(self.X0, self.Y0, **kwargs)
        return self

    def predict(self):
        self.Yh1 = self.clf.predict(self.X1)
        return self

    def score_aic(self, k):
        """
        k is int - # of free parameters
        returns the AIC of model fit to training data
        assuming a Gaussian likelihood
            i.e., assumes errors are normally distributed
        """
        assert 2*np.sqrt(len(self.Y0)) > k
        raise NotImplementedError()
        resid = lambda a, b: ((a-b)**2).sum()
        rss = resid(self.Y0, self.clf.predict(self.X0))
        return len(self.Y0)*np.log(rss/len(self.Y0)) + 2*k

    def score(self, verbose=True):
        """
        calculates the r-squared of model fit to test data
            i.e., 1 - rss/mss
        """
        self.rsq = self.clf.score(self.X1, self.Y1)
        # self.aic = self.score_aic(1)
        if verbose:
            self.print_score()
        return self

    def print_score(self):
        print '{0} score = {1}'.format(self.label, self.rsq)

class OLS(Fit):
    def init_clf(self):
        return sklearn.linear_model.LinearRegression(fit_intercept=self.fit_intercept, normalize=self.normalize)

class Ridge(Fit):
    """
    Ridge regression with priors on tuning parameters
    """
    def init_clf(self):
        # (clf.alpha_, clf.lambda_)
        return sklearn.linear_model.BayesianRidge(fit_intercept=self.fit_intercept)

class Lasso(Fit):
    """
    Lasso model fit with tuning set by cross-validation
    """
    def init_clf(self):
        # clf.alpha_
        return sklearn.linear_model.LassoCV(fit_intercept=self.fit_intercept, normalize=self.normalize)

class ARD(Fit):
    def init_clf(self):
        # (clf.alpha_, clf.lambda_)
        return sklearn.linear_model.ARDRegression(fit_intercept=self.fit_intercept, normalize=self.normalize)

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
        return sklearn.metrics.r2_score(Y1, self.predict(X1))
        # resid = lambda a, b: ((a-b)**2).sum()
        # return 1 - resid(Y1, self.predict(X1))/resid(Y1, Y1.mean())

class ASD(Fit):
    def __init__(self, *args, **kwargs):
        self.Ds = kwargs.pop('Ds')
        self.Dt = kwargs.pop('Dt', None)
        # print type(ASD)
        # print type(self)
        super(ASD, self).__init__(*args, **kwargs)

    def init_clf(self):
        # (clf.coef_, clf.hyper_, clf.Reg_)
        return ASDClf(self.Ds, self.Dt, fit_intercept=self.fit_intercept)

class ASDClf(object):
    def __init__(self, Ds, Dt=None, fit_intercept=False):
        self.Ds = Ds
        self.Dt = Dt
        self.D = [self.Ds] if Dt is None else [self.Ds, self.Dt]
        self.fit_intercept = fit_intercept

    def center_data(self, X, Y):
        """
        if an intercept will be fit, normalize X and Y by subtracting off their means
        source: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/base.py
        """
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            Y_mean = np.mean(Y, axis=0)
            X = X - X_mean
            Y = Y - Y_mean
        else:
            X_mean = np.zeros(X.shape[1])
            Y_mean = 0. if Y.ndim == 1 else np.zeros(Y.shape[1], dtype=X.dtype)
        return X, Y, X_mean, Y_mean

    def set_intercept(self, X_mean, Y_mean):
        if self.fit_intercept:
            self.intercept_ = Y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0.

    def fit(self, X, Y, theta0=None, maxiters=1000, step=0.01, tol=1e-5):
        X, Y, X_mean, Y_mean = self.center_data(X, Y)
        self.coef_, self.Reg_, self.hyper_ = asd.ASD_inner(X, Y, self.D, theta0=theta0)
        self.set_intercept(X_mean, Y_mean)

    def manual_fit(self, X, Y, coef, Reg, hyper):
        X, Y, X_mean, Y_mean = self.center_data(X, Y)
        self.coef_, self.Reg_, self.hyper_ = coef, Reg, hyper
        self.set_intercept(X_mean, Y_mean)

    def predict(self, X1):
        return X1.dot(self.coef_) + self.intercept_

    def score(self, X1, Y1):
        return sklearn.metrics.r2_score(Y1, self.predict(X1))

class ASDRD(ASD):
    def __init__(self, *args, **kwargs):
        self.asdreg = kwargs.pop('asdreg', None)
        super(ASDRD, self).__init__(*args, **kwargs)

    def init_clf(self):
        return ASDRDClf(self.Ds, self.Dt, self.asdreg, fit_intercept=self.fit_intercept)

class ASDRDClf(ASDClf):
    def __init__(self, Ds, Dt=None, asdreg=None, fit_intercept=False):
        self.Ds = Ds
        self.Dt = Dt
        self.D = [self.Ds] if Dt is None else [self.Ds, self.Dt]
        self.asdreg = asdreg
        self.fit_intercept = fit_intercept

    def fit(self, X, Y, theta0=None, maxiters=10000, step=0.01, tol=1e-6):
        X, Y, X_mean, Y_mean = self.center_data(X, Y)
        if self.asdreg is None:
            self.asd_coef_, self.asdreg, self.asd_hyper_ = ASD_inner(X, Y, self.D, theta0=theta0)
            print "ASD complete"
        self.coef_, self.invReg_ = ASDRD_inner(X, Y, self.asdreg, ARD)
        self.set_intercept(X_mean, Y_mean)

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
        yield X0, Y0, X1, Y1

if __name__ == '__main__':
    X = np.arange(10,50)
    Y = np.arange(20,60)
    for t,v in kFoldCV(zip(X, Y), 10):
        print t, v
