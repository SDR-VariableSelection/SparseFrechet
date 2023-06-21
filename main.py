#%%
import numpy as np
import scipy.linalg as la
from scipy.linalg import norm
from scipy.linalg import eigh
from scipy.linalg import sqrtm
from scipy import stats
from joblib import Parallel, delayed
from _functions import *

import sys
n = 2000 #int(sys.argv[1])
p = 3000 #int(sys.argv[2])
mod = 1 #int(sys.argv[3])
rho = 0.5 #int(sys.argv[4])

## One simulation
def onerun(id, n, p, mod):
    # general data
    X, y, beta = examples(mod, n, p, id, rho)
    # normalized beta
    beta_true = beta @ sqrtm(la.inv(beta.T @ beta)) 
    # define metric
    if mod == 1:
        metric = "Euclidean" 
    elif mod == 2:
        metric = "Wasserstein"
    elif mod == 3:
        metric = "Geodesic"   
    M, D = wire(X,y,metric)
    
    ## define synthetic response in q-dimensional
    q = 10
    value, vector = eigh(M, subset_by_index=(p-q, p-1))
    eta = np.fliplr(vector)
    phi = value[::-1]
    x0 = (X - np.mean(X, axis=0)) ## n by p
    y0 = - D @ x0 @ eta * 1/phi * 1/(n-1)
    
    ## multiple lasso: LASSO
    # create a pX4 beta matrix
    beta_l0 = lasso_ind(x0,y0[:,:4])
    # determine the structure dimension
    d1 = determine_d(beta_l0,phi[:4])
    # create a pXd beta matrix
    beta_l = lasso_ind(x0,y0[:,:d])
    # criteria for betahat
    out1 = loss(beta_true, beta_l)
    
    ## group lasso: GLASSO
    # create a pX10 beta matrix
    beta_g0 = multi_lasso(x0,y0[:,:10])
    # determine the structure dimension
    d2 = determine_d(beta_g0,phi)
    # create a pXd beta matrix
    beta_g = multi_lasso(x0, y0[:,:d])
    # criteria for betahat
    out2 = loss(beta_true, beta_g)

    ## LLA entry-wise MCP: LLA_E
    # create a pX4 beta matrix 
    beta10 = lasso_onestep(x0,y0[:,:4])
    # determine the structure dimension
    d3 = determine_d(beta10,phi[:4])
    # create a pXd beta matrix
    beta_l1 = lasso_onestep(x0,y0[:,:d])
    # criteria for betahat
    out3 = loss(beta_true, beta_l1)

    ## LLA group-wise MCP: LLA_G
    # create a pX10 beta matrix
    beta10 = mutlti_onestep(x0,y0[:,:10])
    # determine the structure dimension
    d4 = determine_d(beta10,phi)
    # create a pXd beta matrix
    beta_g1 = mutlti_onestep(x0, y0[:,:d])
    # criteria for betahat
    out4 = loss(beta_true, beta_g1) 

    ## LDA group-wise MCP: LDA(0)
    # create a pX10 beta matrix 
    be10 = multitask_concave(x0,y0[:,:10],np.zeros((p,10)))
    # determine the structure dimension
    d5 = determine_d(be10, phi)
    # create a pXd beta matrix
    beta0 = multitask_concave(x0,y0[:,:d],np.zeros((p,d)))
    # criteria for betahat
    out5 = loss(beta_true, beta0)


    ## LDA group-wise MCP: LDA(G)
    # create a pX10 beta matrix 
    be10 = multitask_concave(x0,y0[:,:10],beta_g0)
    # determine the structure dimension
    d6 = determine_d(be10, phi)
    # create a pXd beta matrix
    beta0 = multitask_concave(x0,y0[:,:d],beta_g)
    # criteria for betahat
    out6 = loss(beta_true, beta0)

    out = np.concatenate((d1,out1,d2,out2,
                          d3,out3,d4,out4,
                          d5,out5,d6,out6), axis = None)
    return out


ite = 4
d = 2
output = Parallel(n_jobs=-1, verbose=10)(
        delayed(onerun)(j, n, p, mod)
        for j in range(ite))
out = np.array(output)
result = np.mean(out, axis = 0).reshape((-1,4))
print(result)

fname = f'n{n}_p{p}_M{mod}_rho{int(rho*10)}.txt'
print(fname)
np.savetxt(fname, out, fmt='%1.4e') 