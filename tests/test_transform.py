# -*- coding: utf-8 -*-


import pytest
import numpy as np
import rankorder.transform as trans



@pytest.mark.parametrize(('n_r', 'n_s'), [(4, 3), (5, 8), (14, 52)])
@pytest.mark.parametrize('method', ['min', 'max', 'dense', 'ordinal'])
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
@pytest.mark.parametrize('method', ['min', 'max', 'dense', 'ordinal'])
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
@pytest.mark.parametrize('method', ['min', 'max', 'dense', 'ordinal'])
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
