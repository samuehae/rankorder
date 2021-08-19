# -*- coding: utf-8 -*-


'''utility module for package rankorder.'''



import numpy as np


def rankdata(a, method):
    '''assigns ranks to values dealing appropriately with ties. 
    
    ranks start at zero and increase with increasing value. 
    the value np.nan is assigned the highest rank.
    
    Parameters
    ----------
    a : array-like
        values to be ranked
    method : {'ordinal', 'random'}
        ranking method to break ties
        'ordinal': all values are given a distinct rank. ties 
            are resolved according to their position in the array.
        'random': like 'ordinal' but in the case of ties the ranks 
            are randomly ordered.
    '''
    
    # implementation is inspired by scipy.stats.rankdata
    
    # check if method is valid
    if method not in ('ordinal', 'random'):
        raise ValueError('unknown method "{}"'.format(method))
    
    
    # convert array-like to array
    a = np.asarray(a)
    
    if method == 'random':
        # randomly permute elements
        # then continue as for method 'ordinal'
        perm = np.random.permutation(a.size)
        a = a[perm]
    
    
    # construct sorting permutation with stable algorithm
    # meaning that order of ties are kept
    sorter = np.argsort(a, kind='mergesort')
    
    # ranks of data is inverse permutation ranks = sorter^{-1} perm
    ranks = np.empty(sorter.size, dtype=np.intp)
    ranks[sorter] = np.arange(sorter.size, dtype=np.intp)
    
    
    if method == 'random':
        # inversely permute rank elements to undo random permutation
        inv = np.argsort(perm, kind='mergesort')
        ranks = ranks[inv]
    
    
    return ranks
