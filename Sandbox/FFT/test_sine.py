# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import leastsq
import pylab as plt
import pandas as pd

#N = 1000 # number of data points
#t = np.linspace(0, 4*np.pi, N)
#data = 3.0*np.sin(t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise

all_data = pd.read_excel('dataLong.xlsx')

data = all_data['x']
t = all_data['time']
t = np.arange(len(t))

guess_mean = np.mean(data)
guess_std = 3000#3*np.std(data)/(2**0.5)
guess_phase = 0
guess_freq = 0.25

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_std*np.sin(guess_freq*t+guess_phase) + guess_mean

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0]*np.sin(x[3]*t+x[1]) + x[2] - data
est_std, est_phase, est_mean, est_freq = leastsq(optimize_func, [guess_std, guess_phase, guess_mean,guess_freq])[0]

# recreate the fitted curve using the optimized parameters
data_fit = est_std*np.sin(est_freq*t+est_phase) + est_mean

plt.plot(data, '.')
plt.plot(data_fit, label='after fitting')
plt.plot(data_first_guess, label='first guess')
plt.legend()
plt.show()