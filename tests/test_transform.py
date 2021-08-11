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
