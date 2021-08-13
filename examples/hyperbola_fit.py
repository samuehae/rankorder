# -*- coding: utf-8 -*-


from __future__ import print_function

import numpy as np
from rankorder import fitting
from scipy.optimize import minimize

import matplotlib.pyplot as pp



'''demonstration of fitting data using the universal rank-order method. 

example function: hyperbola
f(x, a) = a / (x + 1)

the data follows a hyperbola function with additional random noise and an 
additive offset. although the offset is neglected in the fitting function 
the method successfully determines the parameter a. this results from the 
fact that the method only considers rank information of the residuals and 
disregards absolute values.

hint: experiment with the level of the random noise and observe the effect 
on the minimum in the error Q_rms versus parameter a. the minimum becomes 
sharper for smaller noise levels, which is a general feature of the method.
'''



def func(x, a):
    '''hyperbola function.'''
    
    return a / (x + 1.0)




# exact values of fit parameters
params_exact = (2.0, )


# initial values for fitting
params_init = (1.8, )


# additive offset
offset = 1.0



# number of measurement repetitions
n_r = 35

# number of sampling points
n_s = 30



# independent and dependent variable
x_data = np.linspace(0, 1, num=n_s)
y_data = func(x_data, *params_exact) + offset


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
axs[0].plot(x_data, y_data.T, '.', c='tab:Green', alpha=0.1)
axs[0].plot(x_data, func(x_data, *res['x']), c='k', label='fit')
axs[0].plot(x_data, func(x_data, *params_init), ls='dashed', c='k', label='initial')


# plot residuals for each repetition separately
axs[1].plot(x_data, residuals.T, '.', c='tab:Green', alpha=0.3)


# annotate axes
axs[0].set_ylabel('variable y')
axs[1].set_ylabel('residuals')
axs[1].set_xlabel('variable x')

axs[0].legend(frameon=False)



# calculate and plot error measure versus parameter a
a = np.linspace(1.8, 2.2, num=71)
error = [err_func((v, )) for v in a]


fig, ax = pp.subplots()

ax.plot(a, error, '.', c='tab:Orange')

ax.set_xlabel('parameter a')
ax.set_ylabel(u'error $Q_{rms}$')

pp.show()
