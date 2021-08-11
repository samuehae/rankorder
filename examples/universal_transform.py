# -*- coding: utf-8 -*-


import numpy as np
from rankorder import transform

import matplotlib as mpl
import matplotlib.pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages



'''demonstration of the universal rank-order transform. 

the transform is illustrated by data following a linear function 
with different slopes and random noise. the same noise realization 
is used for all slopes to highlight the method.
'''


def create_plot(figsize, gridspec_kw):
    '''create figure and axes for plotting matrices.'''
    
    # create figure and gridspec for subplots
    fig = pp.figure(figsize=figsize)
    gs = mpl.gridspec.GridSpec(**gridspec_kw)
    
    # create subplots to plot matrices
    axs = (
        fig.add_subplot(gs[1, 0]), # for matrix A
        fig.add_subplot(gs[1, 1]), # for matrix R
        fig.add_subplot(gs[4, 0]), # for matrix P
        fig.add_subplot(gs[4, 1]), # for matrix Q
    )
    
    # create subplots for corresponding color bars
    caxs = (
        fig.add_subplot(gs[0, 0]), # for matrix A
        fig.add_subplot(gs[0, 1]), # for matrix R
        fig.add_subplot(gs[3, 0]), # for matrix P
        fig.add_subplot(gs[3, 1]), # for matrix Q
    )
    
    
    # all subplots share x axes
    for ax in axs[1:]:
        axs[0].get_shared_x_axes().join(axs[0], ax)
    
    # subplots in same row (side by side) share y axis
    axs[0].get_shared_y_axes().join(axs[0], axs[1])
    axs[2].get_shared_y_axes().join(axs[2], axs[3])
    
    
    return fig, axs, caxs



def plot_matrices(axs, caxs, x_vals, y_vals, matrices, kwargs):
    '''plot matrices with pcolormesh.'''
    
    # iterate through all matrices with corresponding subplots 
    # and keyword arguments for pcolormesh
    for ax, cax, x, y, matrix, kw in zip(axs, caxs, x_vals, y_vals, matrices, kwargs):
        
        # plot matrix and create corresponding color bar
        ax.pcolormesh(x, y, matrix, **kw)
        pp.colorbar(mappable=ax.collections[0], cax=cax, orientation='horizontal')
    
    
    
    # label color bars and x and y axes
    for cax, clabel in zip(caxs, ['A', 'R', 'P', 'Q']):
        cax.set_ylabel(clabel)
    
    for ax in axs:
        ax.set_xlabel('samples')
    
    axs[0].set_ylabel('repetitions')
    axs[2].set_ylabel('rank')



def calculate_matrices(A):
    '''calculate rank matrix, population matrix and Q matrix.'''
    
    # extract shape of data matrix
    n_r, n_s = A.shape
    
    # calculate rank matrix, population matrix and Q matrix
    R = transform.r_matrix(A)
    P = transform.p_matrix(R)
    Q = transform.q_matrix(P, n_r)
    
    return A, R, P, Q




# number of repetitions
n_r = 31

# number of sampling points
n_s = 21


# sampling points
x = np.linspace(0, 1, num=n_s)

# noise realization for all repetitions and sampling points
noise = np.random.normal(scale=1, size=(n_r, n_s))


# slopes of linear function
slopes = np.linspace(-10, 10, num=11)


# x and y values for matrix plots (matrices A, R, P and Q)
x_vals = [x, x, x, x[:-1]]
y_vals = [np.arange(n_r), np.arange(n_r), np.arange(n_s), np.arange(n_s - 1)]


# keyword arguments for pcolormesh for matrices A, R, P and Q
kwargs = [
    dict(cmap='PRGn', rasterized=True, shading='nearest', vmin=-10, vmax=10), 
    dict(cmap='Blues', rasterized=True, shading='nearest'), 
    dict(cmap='Blues', rasterized=True, shading='nearest', vmin=0, vmax=n_s-1), 
    dict(cmap='BrBG', rasterized=True, shading='nearest', vmin=-2, vmax=2), 
]


# parameters for plots
figsize = (6.4, 7.57)

gridspec_kw = dict(
    nrows=5, ncols=2, 
    bottom=0.07, top=0.92, 
    left=0.1, right=0.9, 
    height_ratios=[1, 10, 0.3, 1, 10], 
    hspace=0.3, 
)



# open pdf file
pdf_pages = PdfPages('universal_transform.pdf')

for slope in slopes:
    
    # calculate data matrix: linear function with random noise
    y = slope * x
    A = y + noise
    
    # obtain matrices A, R, P and Q
    matrices = calculate_matrices(A)
    
    # plot matrices
    fig, axs, caxs = create_plot(figsize, gridspec_kw)
    plot_matrices(axs, caxs, x_vals, y_vals, matrices, kwargs)
    
    for ax in axs:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    
    # revert y axis to obtain indexing for matrix
    # only revert two axis since the others are shared
    axs[0].invert_yaxis()
    axs[2].invert_yaxis()
    
    fig.suptitle('universal transform: y = {:2.1f} x'.format(slope))
    
    
    # save and close figure
    pdf_pages.savefig(fig, dpi=300)
    pp.close(fig)


# close pdf file
pdf_pages.close()
