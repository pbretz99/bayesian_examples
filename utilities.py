# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:24:01 2021

@title: Utilities
@author: Philip Bretz
"""

import numpy as np
import pandas as pd

import kalman_filter as kf

def one_d_kalman_df(theta, Y, fit_values, fit_var):
    #df = pd.DataFrame(np.stack((theta[::,0], Y[::,0]), axis=1),
    #              columns=['State', 'Observations'])
    df = pd.DataFrame(theta[::,0], columns=['State'])
    df['Observations'] = Y[::,0]
    df['Filter'] = fit_values[::,0]
    df['Upper'] = fit_values[::, 0] + 2*np.sqrt(fit_var[::, 0, 0])
    df['Lower'] = fit_values[::, 0] - 2*np.sqrt(fit_var[::, 0, 0])
    return df

def dist_mat(first, second):
    # dist[i,j] is the distance between i-th of first and j-th of second
    N, M = first[0].size, second[0].size
    dist = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            x_1, y_1 = first[0][i], first[1][i]
            x_2, y_2 = second[0][j], second[1][j]
            dist[i, j] = np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
    return dist

def make_grids(dx, side_length, observations):
    size = round(side_length / dx)
    N = size**2
    x_state, y_state = np.meshgrid(np.linspace(0, side_length, size), np.linspace(0, side_length, size))
    x_state, y_state = x_state.reshape(N), y_state.reshape(N)
    x_obs, y_obs = observations[::, 0], observations[::, 1]
    state_grid, obs_grid = [x_state, y_state], [x_obs, y_obs]
    return state_grid, obs_grid

def target_model(params):
    # Parmaters: gamma, var_Y, var_x, var_v, prior_mean, prior_var_x, prior_var_v
    gamma = params[0]
    var_Y = params[1]
    var_x, var_v = params[2], params[3]
    prior_mean, prior_var_x, prior_var_v = params[4], params[5], params[6]
    I_2 = np.identity(2)
    Zero_2 = np.zeros((2, 2))
    top = np.concatenate((I_2, I_2), axis=1)
    bottom = np.concatenate((Zero_2, np.array([[1, -gamma], [-gamma, 1]])), axis=1)
    G = np.concatenate((top, bottom))
    F = np.concatenate((I_2, Zero_2), axis=1)
    top = np.concatenate((var_x*I_2, Zero_2), axis=1)
    bottom = np.concatenate((Zero_2, var_v*I_2), axis=1)
    W = np.concatenate((top, bottom))
    V = var_Y*I_2
    top = np.concatenate((prior_var_x*I_2, Zero_2), axis=1)
    bottom = np.concatenate((Zero_2, prior_var_v*I_2), axis=1)
    prior_var = np.concatenate((top, bottom))
    model = kf.DLM([G, W], [F, V], [prior_mean, prior_var], np.array([0, 0, 0, 0]))
    return model