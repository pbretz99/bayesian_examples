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

import example_models as ex
import kalman_filter as kf
import plotting as pl

active = True
while active:
    default = input('Use default simulation parameters? Y/N: ')
    if default != 'Y':
        params, init = [0.0]*5, [0.0]*2
        W_x = float(input('Enter state position variance: '))
        W_v = float(input('Enter state velocity variance: '))
        V = float(input('Enter space (obervational) variance: '))
        init[0] = float(input('Enter initial position: '))
        init[1] = float(input('Enter initial velocity: '))
        prior_x = float(input('Enter prior guess for position: '))
        prior_v = float(input('Enter prior guess for velocity: '))
        prior_x_var = float(input('Enter prior position variance: '))
        prior_v_var = float(input('Enter prior velocity variance: '))
        obs_params = [np.array([[1, 0]]), np.array([[V]])]
        state_params = [np.array([[1, 1], [0, 1]]), np.array([[W_x, 0], [0, W_v]])]
        prior = [np.array([prior_x, prior_v]), np.array([[prior_x_var, 0], [0, prior_v_var]])]
        model = kf.DLM(state_params, obs_params, prior, np.array([0, 0]))
    else:
        model = ex.default_RWWN_vel_model()
        init = [80, 1]
    length = int(input('Enter length of simulation: '))
    print('Simulating data...')
    theta, Y = model.simulate(length, np.array([init]))
    print('Applying filter to simulated data...')
    fit_values, fit_var = model.kfilter(Y[::,0])
    print('Plotting results...')
    pl.one_d_kalman_plot(theta, Y, fit_values, fit_var)
    again = input('Run program again? Y/N: ')
    if again != 'Y':
        break