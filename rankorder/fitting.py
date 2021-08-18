# -*- coding: utf-8 -*-


'''general regression method based on universal rank-order transform.

method presented in the following publication: 
G. Ierley and A. Kostinski. Phys Rev. X 9, 031039
'''



import numpy as np
from rankorder import transform


def q_matrix_fit(func, params, xdata, ydata, weights=None, method='ordinal'):
    '''calculate Q matrix from residual matrix.
    
    residuals are obtained by subtracting model predictions 
    from each repetition in the data matrix.
    
    Parameters
    ----------
    func : callable
        model function func(xdata, *params)
    params : iterable
        values for fit parameters
    xdata : array
        independent variable with shape (n_s, )
    ydata : array
        dependent variable for each repetition and 
        sampling point with shape (n_r, n_s). 
        note: missing values are not supported
    weights : array
        weights multiplied to residuals, e.g. 1/sigma.
        weights require shape (n_s, ) or (n_r, n_s)
    method : {'min', 'max', 'dense', 'ordinal'}
        ranking method implemented in scipy.stats.rankdata
    '''
    
    # extract shape of data matrix
    n_r, n_s = ydata.shape
    
    # function prediction with shape (n_s, )
    y = func(xdata, *params)
    
    # residuals with shape (n_r, n_s)
    if weights is None:
        residuals = ydata - y
    else:
        residuals = weights * (ydata - y)
    
    # calculate Q matrix from residual matrix
    return transform.data_to_q_matrix(residuals, method)



def q_rms_fit(func, params, xdata, ydata, weights=None, method='ordinal'):
    '''calculate root mean square (rms) of the Q matrix of the residuals.
    
    this value measures the error of the regression method and needs to 
    be minimized by optimizing the parameter values params.
    
    Parameters
    ----------
    func : callable
        model function func(xdata, *params)
    params : iterable
        values for fit parameters
    xdata : array
        independent variable with shape (n_s, )
    ydata : array
        dependent variable for each repetition and 
        sampling point with shape (n_r, n_s). 
        note: missing values are not supported
    weights : array
        weights multiplied to residuals, e.g. 1/sigma.
        weights require shape (n_s, ) or (n_r, n_s)
    method : {'min', 'max', 'dense', 'ordinal'}
        ranking method implemented in scipy.stats.rankdata
    '''
    
    # calculate Q matrix of residuals
    Q = q_matrix_fit(func, params, xdata, ydata, weights, method)
    
    # return root mean square value of elements of matrix Q
    return np.sqrt(np.mean(Q**2))
