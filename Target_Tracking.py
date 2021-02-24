# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:52:52 2021

@title: Bayesian Target Tracking
@author: Philip Bretz
"""

'''
The code below simulates a 2-d random walk with 
time-varying velocity and noise. The Kalman filter 
is then applied to the simulated data.
'''

import numpy as np

import animation as anim
import example_models as ex
#import kalman_filter as kf
import plotting as pl
import utilities as ut

active = True
while active:
    default = input('Use default simulation parameters? Y/N: ')
    if default != 'Y':
        prior_mean, init = [0.0]*4, [0.0]*4
        var_x = float(input('Enter state position variance: '))
        var_v = float(input('Enter state velocity variance: '))
        var_Y = float(input('Enter space (obervational) variance: '))
        init[0] = float(input('Enter initial x position: '))
        init[1] = float(input('Enter initial y position: '))
        init[2] = float(input('Enter initial x velocity: '))
        init[3] = float(input('Enter initial y velocity: '))
        gamma = float(input('Enter drag coefficient gamma: '))
        prior_mean[0] = float(input('Enter prior guess for x position: '))
        prior_mean[1] = float(input('Enter prior guess for y position: '))
        prior_mean[2] = float(input('Enter prior guess for x velocity: '))
        prior_mean[3] = float(input('Enter prior guess for y velocity: '))
        prior_var_x = float(input('Enter prior position variance: '))
        prior_var_v = float(input('Enter prior velocity variance: '))
        params = [gamma, var_Y, var_x, var_v, np.array(prior_mean), prior_var_x, prior_var_v]
    else:
        params = ex.default_target_params()
        init = [0, 0, 1, 1]
    model = ut.target_model(params)
    length = int(input('Enter length of simulation: '))
    print('Simulating data...')
    theta, Y = model.simulate(length, np.array([init]))
    print('Applying filter to simulated data...')
    fit_values, fit_var = model.kfilter(Y[::,0:2])
    print('Plotting results...')
    pl.target_plot(Y, fit_values)
    print('Animating results...')
    anim.animate_target(Y, fit_values, 'Target Tracking', 'target_tracking.gif')
    again = input('Run program again? Y/N: ')
    if again != 'Y':
        break