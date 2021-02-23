# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:52:52 2021

@title: Random Walk With Noise
@author: Philip Bretz
"""

'''
The code below simulates a random walk with 
noise. The Kalman filter is then applied to 
the simulated data.
'''

import numpy as np

import kalman_filter as kf
import plotting as pl
import utilities as ut

active = True
while active:
    default = input('Use default simulation parameters? Y/N: ')
    if default != 'Y':
        params = [0.0]*5
        text = 'Enter '
        params[0] = float(input(text+'state (movement) variance: '))
        params[1] = float(input(text+'space (obervational) variance: '))
        init = float(input(text+'initial state value: '))
        params[2] = float(input(text+'prior mean: '))
        params[3] = float(input(text+'prior variance: '))
        length = int(input(text+'length of simulation: '))
        space_params = [np.array([1]), np.array([[params[1]]])]
        state_params = [np.array([1]), np.array([[params[0]]])]
        prior = [np.array([params[2]]), np.array([[params[3]]])]
        source = np.array([0])
        model = kf.DLM(state_params, space_params, prior, source)
    else:
        model = ut.example_model()
        init = 80
        length = 75
    print('Simulating data...')
    theta, Y = model.simulate(length, np.array([init]))
    print('Applying filter to simulated data...')
    fit_values, fit_var = model.kfilter(Y[::,0])
    print('Plotting results...')
    pl.one_d_kalman_plot(theta, Y, fit_values, fit_var)
    again = input('Run program again? Y/N: ')
    if again != 'Y':
        break