import numpy as np
import pandas as pd

import scipy.linalg as la
from scipy.linalg import norm
from scipy.linalg import eigh
from scipy.linalg import sqrtm
from scipy import stats
from sklearn.model_selection import KFold # import KFold
from sklearn.utils import resample
import random
from joblib import Parallel, delayed
from sklearn import linear_model
import matplotlib.pyplot as plt
import pywt
from sklearn.cluster import KMeans

def examples(model, n, p, seedid, rho = 0.5):
    # initialize the pseudo-random number generator 
    rng = np.random.RandomState(seedid)
    # define a Toeplitz matrix
    covx = la.toeplitz(rho ** np.arange(p))
    # generate multivariate normal random variables
    X = rng.multivariate_normal(np.zeros(p), covx, size = n)
    # structure dimension
    d = 2
    # true directions in central subspace
    beta = np.zeros((p,d))
    beta[:5, 0] = 1
    beta[5:10, 1] = 1

    if model == 1:
        y = np.zeros((n, 3))
        x1 = (X @ beta[:,0])
        x2 = (X @ beta[:,1])
        y[:,0] = 1 + x1  + np.random.randn(n)
        y[:,1] = x2 + np.random.randn(n)
        y[:,2] = np.abs(x1) * np.random.randn(n)
    elif model == 2:
        # points per distribution
        L = 100
        mu = (X @ beta[:,0])
        sd = (X @ beta[:,1])
        y = np.zeros((n, L-1))
        normppf = np.zeros(L-1)
        normppf = stats.norm.ppf(np.arange(1,L,1)/L)

        for i in range(n):
            y[i,:] = mu[i] + sd[i]*normppf
    elif model == 3:
        y = np.zeros((n, 4))
        u = np.random.randn(n) * 0.1
        v = 0.2
        a1 = np.sin(v * (X+1) @ beta[:,0])
        a2 = np.cos(v * (X+1) @ beta[:,0])
        b1 = np.sin(v * (X+1) @ beta[:,1])
        b2 = np.cos(v * (X+1) @ beta[:,1])
        y[:,0] = np.cos(u) * a1 * b1
        y[:,1] = np.cos(u) * a1 * b2
        y[:,2] = np.cos(u) * a2
        y[:,3] = np.sin(u)        

    return X, y, beta


def wire(X, y, metric):   
    """calculate sample kernel matrix for WIRE
    X: design matrix
    y: response vector
    metric: user-defined metric
    """ 
    n = X.shape[0]
    y = y.reshape((n,-1))
    x0 = X - np.mean(X, axis=0)            
    q = y.shape[1]
    if metric == "Euclidean":
        temp = np.zeros((n, n, q))
        for i in range(q):
            temp[:,:,i] = np.subtract(y[:,i].reshape(n,1), y[:,i].reshape(1,n))
        D = np.sqrt(np.sum(np.square(temp),2))
    if metric == "Wasserstein":
        temp = np.zeros((n, n, q))
        for i in range(q):
            temp[:,:,i] = np.subtract(y[:,i].reshape(n,1), y[:,i].reshape(1,n))
        D = np.sqrt(np.mean(np.square(temp),2))
#         indnon0 = np.nonzero(D)
#         D = D/(1+D)
    if metric == "Geodesic":
        D = np.arccos(np.minimum(y@y.T, 1)) #np.arccos(y @ y.T)
    M = -(x0.T @ D @ x0)/(n**2)
    return M, D

def loss(A, A_hat):
    '''assessment of estimates
    A: true parameter
    A_hat: estimate
    '''
    loss_general = norm(A_hat @ la.inv(A_hat.T @ A_hat) @ A_hat.T - A @ la.inv(A.T @ A) @ A.T)
    S = np.nonzero(norm(A, axis = 1))[0]
    S_hat = np.nonzero(norm(A_hat, axis = 1))[0]
    false_positive = len(set(S_hat).difference(S)) ## false positive
    false_negative = len(set(S).difference(S_hat)) ## false negative

    return loss_general, false_positive, false_negative

def estimate_d(X, y, metric, ker):
    M,_ = ker(X,y,metric)
    n,p = X.shape
    sigma = np.cov(X.T)
    k = 10
    value, beta_hat = eigh(M, subset_by_index=(p-k, p-1))
    beta_hat = np.fliplr(beta_hat)
    value = value[::-1]
    p2 = value/(1+np.sum(value))
    fk = np.zeros(k)
    nbs = 50

    def one(j):
        X_bs, y_bs = resample(X, y, replace=True)
        sigma_bs = np.cov(X_bs.T)
        M_bs,_ = ker(X_bs, y_bs, metric)
        value_b, beta_b = eigh(M_bs, subset_by_index=(p-k, p-1))
        beta_b = np.fliplr(beta_b)
        value_b = value_b[::-1]
        fk = np.zeros(k)
        for i in range(k-1):
            fk[i+1] = 1-np.abs(la.det(beta_hat[:,0:(i+1)].T @ beta_b[:,0:(i+1)]))
        return fk

    for _ in range(nbs):    
        X_bs, y_bs = resample(X, y, replace=True)
        sigma_bs = np.cov(X_bs.T)
        M_bs,_ = wire(X_bs, y_bs, metric)
        value_b, beta_b = eigh(M_bs, subset_by_index=(p-k, p-1))
        beta_b = np.fliplr(beta_b)
        value_b = value_b[::-1]
        for i in range(k-1):
            fk[i+1] += (1-np.abs(la.det(beta_hat[:,0:(i+1)].T @ beta_b[:,0:(i+1)])))/nbs
    
    gn = fk/(1+np.sum(fk)) + value/(1+np.sum(value))
    d = np.argmin(gn)
    return d

def pho(t, lam, eta=2):
    t = np.abs(t)    
    index = (t < lam * eta)
    out = np.zeros(len(t))
    out[index] = (lam*t - t**2/(2*eta))[index]
    out[t >= lam * eta] = (lam**2 * eta/2)
    return out
    
def pho_prime(t, lam, eta=2):
    t = np.abs(t)
    return np.maximum(lam - t/eta, 0)

def gradiant_f(beta, x, y):
    n= x.shape[0]
    return - 1/n * x.T @ (y - x @ beta)

def hessian_f(x):
    n = x.shape[0]
    return 1/n * x.T @ x

def object(x, y, beta,lam, eta):
    n = x.shape[0]
    rowbeta = norm(beta, axis = 1)
#     penalty = [pho(t, lam, eta) for t in rowbeta]
    penalty = pho(rowbeta, lam, eta)
    return 1/(2*n) * norm(y - x @ beta) **2 + np.sum(penalty)

def determine_d(betas, phi):
    temp = phi * norm(betas, axis = 0)
    kmeans = KMeans(init="random",
                    n_clusters=2,
                    n_init=20,
                    max_iter=300)
    kmeans.fit(np.array(temp).reshape(-1,1))
    return sum(kmeans.labels_==kmeans.labels_[0])

def lasso_ind(x0,y0):
    p = x0.shape[1]
    q = y0.shape[1]

    beta_hat = np.zeros((p,q))
    for i in range(q):
        clf = linear_model.LassoCV(fit_intercept = False)
        clf.fit(X=x0, y=np.ravel(y0[:,i]))
        beta_hat[:,i] = clf.coef_
    return beta_hat

def lasso_onestep(x0,y0):
    p = x0.shape[1]
    q = y0.shape[1]

    beta_hat = np.zeros((p,q))
    for i in range(q):
        clf0 = linear_model.LassoCV(fit_intercept = False)
        clf0.fit(X=x0, y=np.ravel(y0[:,i]))
        lam = clf0.alpha_

        gprime = lambda w:  np.maximum(1- np.abs(w)/(lam*2), 0) + np.finfo(float).eps
        weights = gprime(clf0.coef_)
        X_w = x0 / weights
        clf = linear_model.LassoCV(fit_intercept = False)
        clf.fit(X_w, np.ravel(y0[:,i]))
        beta_hat[:,i] = clf.coef_ / weights
    
    return beta_hat


def multi_lasso(x0,y0):
    p = x0.shape[1]
    d = y0.shape[1]
    
    clf = linear_model.MultiTaskElasticNetCV(l1_ratio=1, fit_intercept=False)
    clf.fit(X=x0, y=y0)
    beta_hat = clf.coef_.T
    return beta_hat

def mutlti_onestep(x0,y0):
    p = x0.shape[1]
    d = y0.shape[1]

    clf = linear_model.MultiTaskElasticNetCV(l1_ratio=1, fit_intercept=False)
    clf.fit(X=x0, y=y0)

    beta_hat = clf.coef_.T
    coef_norm = norm(beta_hat,axis = 1)
    lam = clf.alpha_
    gprime = lambda w:  np.maximum(1- np.abs(w)/(lam*2), 0) + np.finfo(float).eps
    weights = gprime(coef_norm)
    X_w = x0 / weights[np.newaxis, :]
    clf = linear_model.MultiTaskElasticNetCV(l1_ratio=1, fit_intercept=False)
    clf.fit(X=X_w, y=y0)
    beta_hat = (clf.coef_/ weights).T
    return beta_hat

def lasso_concave(x0,y0,init=None):
    n, p = x0.shape
    d = y0.shape[1]
    if init is None:
        init = lasso_ind(x0,y0)
    L = norm(1/n * x0.T @ x0, 2)
    lam_max = np.max(np.abs(x0.T @ y0/n))
    lam = lam_max/6
    beta_old = init #np.zeros((p,d))
    max_ite = 100
    obj_seq = np.zeros(max_ite)
    for i in range(max_ite):
        temp = beta_old - gradiant_f(beta_old, x0, y0)/L
        pp = pho_prime(beta_old, lam)
        assert np.allclose(pp.shape, temp.shape)
        beta_new = pywt.threshold(temp, pp/L, 'soft')
        obj_seq[i] = object(x0, y0, beta_new, lam, eta=2)
        beta_old = beta_new
        if obj_seq[i] > obj_seq[i-1] and i > 0:
            break
    return beta_old

def multitask_concave(x0,y0,init=None):
    n, p = x0.shape
    d = y0.shape[1]
    if init is None:
        init = multi_lasso(x0,y0)
    L = norm(1/n * x0.T @ x0, 2)
    lam_max = max(norm(x0.T @ y0/n, axis = 1))
    lam = lam_max/6
    beta_old = init #np.zeros((p,d))
    max_ite = 100
    obj_seq = np.zeros(max_ite)
    for i in range(max_ite):
        temp = beta_old - gradiant_f(beta_old, x0, y0)/L
        rowbetan = norm(beta_old, axis = 1)
        pp = pho_prime(rowbetan, lam)
        coef = np.maximum(1 - pp/(L * norm(temp, axis = 1)), 0)
        beta_new = (coef * temp.T).T
        obj_seq[i] = object(x0, y0, beta_new, lam, eta=2)
        beta_old = beta_new
#     plt.plot(obj_seq)
    return beta_old