import numpy as np
import sklearn.cross_validation
import sklearn.linear_model

rmse = lambda Y, Yh: np.sqrt(np.mean((Yh - Y) ** 2))
def rmse_clf(clf, X, Y):
    return rmse(clf.predict(X), Y)

def ols(X, Y):
    return np.linalg.lstsq(X, Y)[0]

def ols2(X, Y):
    clf = sklearn.linear_model.LinearRegression(fit_intercept=False, normalize=False)
    clf.fit(X, Y)
    return clf.coef_

def bilinear(X, Y, niters=1000):
    whs = ols(np.sum(X, 1), Y)
    for _ in xrange(niters):
        wht = ols(X.dot(whs), Y)
        whs = ols(wht.dot(X), Y)
    return whs, wht

def ridge(X, Y, alphas=[0.1, 1.0, 10.0]):
    clf = sklearn.linear_model.RidgeCV(alphas=alphas, fit_intercept=False, normalize=False)
    clf.fit(X, Y)
    return clf.coef_, clf.alpha_

def lasso(X, Y):
    clf = sklearn.linear_model.LassoCV(alphas=alphas, fit_intercept=False, normalize=False)
    clf.fit(X, Y)
    return clf.coef_, clf.alpha_

def predict(X, w1, w2=None):
    if w2 is None:
        if len(w1.shape) == 1 or w1.shape[1] == 1:
            if X.shape[-1] == w1.shape[0]:
                return X.dot(w1)
            elif X.shape[0] == w1.shape[0]:
                return w1.dot(X)
        elif (X.shape[-1] == w1.shape[-1]) and (X.shape[-2] == w1.shape[-2]):
            return np.einsum('abc,bc -> a', X, w1)
    elif X.shape[-1] == w2.shape[0]:
        return w1.dot(X).dot(w2)
    elif X.shape[0] == w2.shape[0]:
        return w2.dot(X).dot(w1)
    err = "Bad shapes ({0} x {1}{2}{3}) = no prediction."
    raise Exception(err.format(X.shape, w1.shape, ' x ' if w2 else '', w2.shape if w2 else ''))

def trainAndTest(X, Y, trainPct=0.9):
    """
    returns (X,Y) split into training and testing sets
    """
    X0, X1, Y0, Y1 = sklearn.cross_validation(X, Y, trainPct)
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
