# -*- coding: utf-8 -*-


from __future__ import print_function

import numpy as np
from rankorder import fitting
from scipy.optimize import minimize

import matplotlib as mpl
import matplotlib.pyplot as pp
from mpl_toolkits.axes_grid1 import inset_locator



'''demonstration of fitting data using the universal rank-order method. 

example function: biexponential
f(x, a, b) = exp(a*x) - exp(b*x)

the error Q_rms versus the fit parameters displays jumps between regions 
of (nearly) constant error. the non-smooth behaviour requires to use 
minimization methods that do not rely on derivatives. in addition, the 
error can exhibit many local minima, which make it challenging to find 
the global minimum.

hint: experiment with the number of repetitions n_r and observe how the 
error landscape changes. more repetitions lead to smaller jumps and thus 
to a smoother landscape.
'''



def func(x, a, b):
    '''biexponential function.'''
    
    return np.exp(a*x) - np.exp(b*x)




# exact values of fit parameters
params_exact = (-2.0, -5.0)


# initial values for fitting
params_init = (-1.0, -4.0)



# number of measurement repetitions
n_r = 8

# number of sampling points
n_s = 30



# independent and dependent variable
x_data = np.linspace(0, 1, num=n_s)
y_data = func(x_data, *params_exact)


# add random noise to data values (simulate measurements)
np.random.seed(2274362)
y_data = y_data + np.random.normal(scale=3e-2, size=(n_r, n_s))




# create error function for fitting
err_func = lambda params: \
    fitting.q_rms_fit(func, params, x_data, y_data)


# minimize error function
res = minimize(err_func, x0=params_init, method='Nelder-Mead', 
    options=dict(xtol=1e-8, ftol=1e-8))

if not res['success']:
    raise Exception('minimization failed during fitting')


# calculate residuals for optimized parameters
residuals = y_data - func(x_data, *res['x'])



# output fit results
print('exact parameters: ', params_exact)
print('initial parameters: ', params_init)
print('fitted parameters: ', res['x'])



# plot fit results
fig, axs = pp.subplots(2, sharex=True)

# plot data for fitting and initial and optimized fit curves
axs[0].plot(x_data, y_data.T, '.', c='tab:Green', alpha=0.5)
axs[0].plot(x_data, func(x_data, *res['x']), c='k', label='fit')
axs[0].plot(x_data, func(x_data, *params_init), ls='dashed', c='k', label='initial')


# plot residuals for each repetition separately
axs[1].plot(x_data, residuals.T, '.', c='tab:Green', alpha=0.8)


# annotate axes
axs[0].set_ylabel('variable y')
axs[1].set_ylabel('residuals')
axs[1].set_xlabel('variable x')

axs[0].legend(frameon=False)



# calculate and plot error measure versus fit parameters
# parameters for main plot
a_main = np.linspace(-1, -3, num=101)
b_main = np.linspace(-4, -6, num=102)

# parameters for inset
a_inset = np.linspace(-2.12, -1.92, num=101)
b_inset = np.linspace(-5.10, -4.80, num=102)



# create error function for plotting
err_func = lambda *params: \
    fitting.q_rms_fit(func, params, x_data, y_data)

err_func = np.vectorize(err_func)


# calculate error measure versus fit parameters
error_main = err_func(a_main, b_main[:, np.newaxis])
error_inset = err_func(a_inset, b_inset[:, np.newaxis])



# plot error measure versus fit parameters
fig, axs = pp.subplots(2, gridspec_kw=dict(height_ratios=[1, 15]))

axs[1].pcolormesh(a_main, b_main, error_main, 
    cmap='viridis', rasterized=True, shading='nearest')

pp.colorbar(mappable=axs[1].collections[0], cax=axs[0], orientation='horizontal')
axs[0].set_ylabel(u'$Q_{rms}$')

# indicate exact and fitted parameters
axs[1].plot(*res['x'], marker='o', c='tab:Orange', alpha=0.3)
axs[1].plot(*params_exact, marker='o', c='tab:Green', alpha=0.3)


# indicate region of inset in main plot
xy = (a_inset.min(), b_inset.min())
width = a_inset.max() - a_inset.min()
height = b_inset.max() - b_inset.min()

rect = mpl.patches.Rectangle(xy, width, height, facecolor='none', 
    edgecolor='0.1', transform=axs[1].transData)

axs[1].add_patch(rect)


# annotate axes and adjust tick locations
axs[1].set_xlabel('parameter a')
axs[1].set_ylabel('parameter b')

axs[1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
axs[1].yaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))



# plot error measure in inset
ax_inset = inset_locator.inset_axes(axs[1], 
    width='40%', height='50%', loc='upper right')
ax_inset.pcolormesh(a_inset, b_inset, error_inset, cmap='viridis', 
    rasterized=True, shading='nearest')

ax_inset.set_yticks([])
ax_inset.set_xticks([])

pp.show()
