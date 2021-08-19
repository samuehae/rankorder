# -*- coding: utf-8 -*-


from __future__ import division

import pytest
import numpy as np
import rankorder.transform as trans



@pytest.mark.parametrize(('n_r', 'n_s'), [(4, 3), (5, 8), (14, 52)])
@pytest.mark.parametrize('method', ['ordinal', 'random'])
@pytest.mark.parametrize('seed', [2274362, None])

def test_matrix_shapes(n_r, n_s, method, seed):
    '''check shapes of matrices A, R, P and Q.'''
    
    # create random data matrix with known seed
    np.random.seed(seed)
    A = np.random.random((n_r, n_s))
    
    # consecutively apply transforms
    R = trans.r_matrix(A, method)
    P = trans.p_matrix(R)
    Q = trans.q_matrix(P, n_r)
    
    
    # compare shapes of matrices with expectation
    assert A.shape == (n_r, n_s)
    assert R.shape == (n_r, n_s)
    assert P.shape == (n_s, n_s)
    assert Q.shape == (n_s - 1, n_s - 1)



@pytest.mark.parametrize(('n_r', 'n_s'), [(4, 3), (5, 8), (14, 52)])
@pytest.mark.parametrize('method', ['ordinal', 'random'])
@pytest.mark.parametrize('seed', [2274362, None])

def test_r_matrix(n_r, n_s, method, seed):
    '''check entries of rank matrix.'''
    
    # create random data matrix with known seed
    np.random.seed(seed)
    A = np.random.random((n_r, n_s))
    
    # calculate rank matrix
    R = trans.r_matrix(A, method)
    
    
    # obtain indices that sort rankings for each repetition separately
    inds = np.argsort(R, axis=1)
    
    # iterate through all repetitions
    for a, ind in zip(A, inds):
        
        # reorder data values according to ranking
        a_ordered = a[ind]
        
        # check that ordered values are sorted
        assert np.all(a_ordered[:-1] <= a_ordered[1:])



@pytest.mark.parametrize(('n_r', 'n_s'), [(4, 3), (5, 8), (14, 52)])
@pytest.mark.parametrize('method', ['ordinal', 'random'])
@pytest.mark.parametrize('seed', [2274362, None])

def test_p_matrix(n_r, n_s, method, seed):
    '''check sum conditions of population matrix.'''
    
    # create random data matrix with known seed
    np.random.seed(seed)
    A = np.random.random((n_r, n_s))
    
    # calculate rank and population matrix
    R = trans.r_matrix(A, method)
    P = trans.p_matrix(R)
    
    
    # check sum conditions of population matrix
    assert np.allclose(np.sum(P, axis=0), n_r)
    assert np.allclose(np.sum(P, axis=1), n_r)



@pytest.mark.parametrize(('n_r', 'n_s'), [(4, 3), (5, 8), (14, 52)])
@pytest.mark.parametrize('seed', [2274362, None])

def test_q_matrix(n_r, n_s, seed):
    '''test faster q transform with slower (more direct) version.'''
    
    # construct random input matrix for q transform with known seed
    # elements in input matrix must sum to n_r in each column and row
    np.random.seed(seed)
    P = np.random.random((n_s-1, n_s-1))
    P = sumified_matrix(P, total=n_r)
    
    # calculate q transform
    Q_slow = q_matrix_slow(P, n_r)
    Q_fast = trans.q_matrix(P, n_r)
    
    
    # compare output of both methods
    assert np.allclose(Q_slow, Q_fast)


def sumified_matrix(a, total):
    '''create matrix where each column (row) sums to the same value.
    
    construction:
    * input matrix a is square with shape (n, n)
    * output matrix b is square with shape (n+1, n+1)
    * matrix b contains matrix a as the upper left block
    * last column and row are filled with sum conditions
    '''
    
    # extract shape of matrix
    n, m = a.shape
    
    # ensure that matrix is square shaped
    # else the sum constraints can not be fulfilled
    if n != m:
        raise Exception('matrix must be square')
    
    
    # allocate space for output matrix
    b = np.empty((n+1, n+1))
    
    # upper left block corresponds to matrix a
    b[:n, :n] = a
    
    # last row follows directly with columnwise sum conditions, 
    # excluding bottom right matrix element
    b[n, :n] = total - np.sum(a, axis=0)
    
    # last column follows directly with rowwise sum conditions, 
    # excluding bottom right matrix element
    b[:n, n] = total - np.sum(a, axis=1)
    
    # calculate bottom right matrix element
    b[n, n] = np.sum(a) - (n-1)*total
    
    return b


def q_matrix_slow(P, n_r, dtype=np.float):
    '''calculates Q matrix from population matrix.'''
    
    # convert array-like to array
    P = np.asarray(P)
    
    # extract shape of population matrix
    n_s, m = P.shape
    
    # ensure that matrix P is square shaped
    if n_s != m:
        raise Exception('matrix P must be square')
    
    
    # allocate space for matrix Q
    Q = np.empty((n_s-1, n_s-1), dtype=dtype)
    
    # calculate matrix Q (see docstring of module transform)
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
