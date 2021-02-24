# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:52:52 2021

@title: Temperature on a Grid
@author: Philip Bretz
"""

'''
The code below simulates time-evolving temperature
on a grid. The Kalman filter is then applied to 
the simulated data.
'''

import animation as anim
import example_models as ex
import kalman_filter as kf
import temp_grid as tg
import utilities as ut

import numpy as np

active = True
while active:
    default_grid = input('Use default grid construction? Y/N: ')
    if default_grid != 'Y':
        side_length = float(input('Enter side length of square room (in meters): '))
        dx = float(input('Enter spacing of state grid (in meters): '))
        print('Now enter locations of (at least 2) thermometers.')
        entering_observers = True
        obs_points = []
        count = 1
        while entering_observers:
            point = [0, 0]
            point[0] = float(input('Enter x-coordinate of thermometer ' + str(count) + ': '))
            point[1] = float(input('Enter y-coordinate of thermometer ' + str(count) + ': '))
            obs_points.append(point)
            count += 1
            more_points = input('Enter another thermometer? Y/N: ')
            if more_points != 'Y':
                break
        print('Now enter locations of heat sources.')
        source_points, source_weights = [], []
        entering_sources = True
        count = 1
        while entering_sources:
            point = [0, 0]
            point[0] = float(input('Enter x-coordinate of source ' + str(count) + ': '))
            point[1] = float(input('Enter y-coordinate of source ' + str(count) + ': '))
            source_points.append(point)
            source_weights.append(1)
            count += 1
            more_points = input('Enter another heat source? Y/N: ')
            if more_points != 'Y':
                break
        print('Now enter locations of heat sinks.')
        entering_sinks = True
        count = 1
        while entering_sinks:
            point = [0, 0]
            point[0] = float(input('Enter x-coordinate of sink ' + str(count) + ': '))
            point[1] = float(input('Enter y-coordinate of sink ' + str(count) + ': '))
            source_points.append(point)
            source_weights.append(-1)
            count += 1
            more_points = input('Enter another heat sink? Y/N: ')
            if more_points != 'Y':
                break
        state_grid, obs_grid = ut.make_grids(dx, side_length, np.array(obs_points))
        state_grid, source_grid = ut.make_grids(dx, side_length, np.array(source_points))
        source_terms = [source_grid, np.array(source_weights)]
    else:
        state_grid, obs_grid, source_terms = ex.default_Heat_grid()
    # Get size of state vector
    N = state_grid[0].size
    # Write option to enter parameters
    default = input('Use default simulation parameters? Y/N: ')
    if default != 'Y':
        params = [0]*4
        init_temp = float(input('Enter initial temperature: '))
        init = np.array([init_temp]*N)
        params[0] = float(input('Enter temperature movement variance: '))
        params[1] = float(input('Enter observational variance: '))
        params[2] = float(input('Enter prior guess of overall temperature: '))
        params[3] = float(input('Enter prior variance: '))
    else:
        params = ex.default_Heat_params()
        init = np.array([70]*N)
    sim_model = tg.temp_model(state_grid, obs_grid, source_terms, params)
    source_in_filter = input('Allow Kalman filter to know about the sources/sinks? Y/N: ')
    if source_in_filter != 'Y':
        pred_model = kf.DLM(sim_model.state_params, sim_model.space_params, sim_model.filter_params, np.array([0]*N))
    else:
        pred_model = sim_model
    length = int(input('Enter desired number of frames to simulate: '))
    print('Simulating data...')
    theta, Y = sim_model.simulate(length, init)
    print('Filtering simulated data...')
    fit_values, fit_var = pred_model.kfilter(Y)
    print('Constructing animations...')
    anim.animate_heat_eqn_wrapper(state_grid, obs_grid, theta, fit_values, Y)
    again = input('Run program again? Y/N: ')
    if again != 'Y':
        break