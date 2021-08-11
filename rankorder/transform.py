# -*- coding: utf-8 -*-


'''universal rank-order transform to analyze noisy data.

method presented in the following publication: 
G. Ierley and A. Kostinski. Phys Rev. X 9, 031039


central quantities of the method: 

* data matrix A[i, k] with shape (n_r, n_s).
  collects time series at sampling points x_k measured 
  for several repetitions labelled by index i.

* rank matrix R[i, k] with shape (n_r, n_s).
  assigns ranks to data values for each repetition i separately.
  ranks are integers from zero to n_s - 1.

* population matrix P[r, k] with shape (n_s, n_s).
  counts occurances of rank r at sampling point x_k.

* matrix Q[j, k] with shape (n_s - 1, n_s - 1).
  constructed from the following partition of matrix P
  {{S2, S1}, {S3, S4}} = {{P[:j+1, :k+1], P[:j+1, k+1:]}, {P[j+1:, :k+1], P[j+1:, k+1:]}}
  
  Q[j, k] = n_s/n_r * [(S2.sum() + S4.sum())/(S2.size + S4.size) - 
    (S1.sum() + S3.sum())/(S1.size + S3.size)]
'''



from __future__ import division

import numpy as np
from builtins import range
from scipy.stats import rankdata


def r_matrix(A, method='ordinal'):
    '''calculates rank matrix from data matrix (see module's docstring).
    
    Parameters
    ----------
    A : array-like
        data matrix A[i, k] with shape (n_r, n_s).
        collects time series at sampling points x_k measured 
        for several repetitions labelled by index i.
    method : {'min', 'max', 'dense', 'ordinal'}
        use only methods from scipy.stats.rankdata that return integers
        note: attributes highest rank to np.nan
    '''
    
    # subtract one to start ranking from zero
    return np.apply_along_axis(rankdata, 1, A, method) - 1



def p_matrix(R):
    '''calculates population matrix from rank matrix (see module's docstring).
    
    Parameters
    ----------
    R : array-like
        rank matrix R[i, k] with shape (n_r, n_s).
        assigns ranks to data values for each repetition i separately.
        ranks are integers from zero to n_s - 1.
    '''
    
    # convert array-like to array
    R = np.asarray(R)
    
    # extract shape of rank matrix
    n_r, n_s = R.shape
    
    return np.apply_along_axis(np.bincount, 0, R, minlength=n_s)



def q_matrix(P, n_r, dtype=np.float):
    '''calculates Q matrix from population matrix (see module's docstring).
    
    Parameters
    ----------
    P : array-like
        population matrix P[r, k] with shape (n_s, n_s).
        counts occurances of rank r at sampling point x_k.
    n_r : scalar
        number of repetitions at each sampling point
    dtype : data-type
        desired data-type of matrix Q
    '''
    
    # convert array-like to array
    P = np.asarray(P)
    
    # extract shape of population matrix
    n_s, m = P.shape
    
    # ensure that matrix P is square shaped
    if n_s != m:
        raise Exception('matrix P must be square')
    
    
    # allocate space for matrix Q
    Q = np.empty((n_s-1, n_s-1), dtype=dtype)
    
    # calculate matrix Q (see module's docstring)
    for j in range(n_s - 1):
        for k in range(n_s - 1):
            
            # partition matrix P
            S1 = P[:j+1, k+1:]
            S2 = P[:j+1, :k+1]
            S3 = P[j+1:, :k+1]
            S4 = P[j+1:, k+1:]
            
            # calculate matrix Q (ensure float division)
            Q[j, k] = n_s / n_r * ((S2.sum() + S4.sum())/(S2.size + S4.size) - \
                (S1.sum() + S3.sum())/(S1.size + S3.size))
    
    return Q



def data_to_q_matrix(A, method='ordinal', dtype=np.float):
    '''calculates Q matrix from data matrix (see module's docstring).
    
    Parameters
    ----------
    A : array-like
        data matrix A[i, k] with shape (n_r, n_s).
        collects time series at sampling points x_k measured 
        for several repetitions labelled by index i.
    method : {'min', 'max', 'dense', 'ordinal'}
        use only methods from scipy.stats.rankdata that return integers
        note: attributes highest rank to np.nan
    dtype : data-type
        desired data-type of matrix Q
    '''
    
    # extract shape of data matrix
    n_r, n_s = A.shape
    
    # calculate Q matrix of data matrix
    R = r_matrix(A, method)
    P = p_matrix(R)
    Q = q_matrix(P, n_r, dtype)
    
    return Q
