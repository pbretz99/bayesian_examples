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
    df = pd.DataFrame(np.stack((theta[::,0], Y[::,0]), axis=1),
                  columns=['State', 'Observations'])
    df['Filter'] = fit_values
    df['Upper'] = fit_values[::, 0] + 2*np.sqrt(fit_var[::, 0, 0])
    df['Lower'] = fit_values[::, 0] - 2*np.sqrt(fit_var[::, 0, 0])
    return df

def example_model():
    # Create default model
    V, W = 0.5, 0.1
    space_params = [np.array([1]), np.array([[V]])]
    state_params = [np.array([1]), np.array([[W]])]
    prior = [np.array([80]), np.array([[1]])]
    source = np.array([0])
    model = kf.DLM(state_params, space_params, prior, source)
    return model

def example_temp_params():
    # Create default model parameters
    diffusion = 5
    movement_cor = 0.5
    movement_var = 15
    observation_loc = 10
    observation_noise = 1
    prior_guess = 70
    prior_var = 1
    params = [diffusion, movement_cor, movement_var, observation_loc, observation_noise, prior_guess, prior_var]
    return params


def temp_model(state_grid, obs_grid, params):
    # Parameters are: diffusion, movement correlation, movement variance, observation locality, observation noise, prior guess, and prior variance
    diffusion = params[0]
    movement_cor, movement_var = params[1], params[2]
    observation_loc, observation_noise = params[3], params[4]
    prior_guess, prior_var = params[5], params[6]
    N, p = state_grid[0].size, obs_grid[0].size
    # Get distance matrices
    dist_state = dist_mat(state_grid, state_grid)
    dist_obs = dist_mat(obs_grid, state_grid)
    # Create model for temperature
    G = np.exp(-dist_state/diffusion)
    for i in range(N):
        G[i,::] = G[i,::] / np.sum(G[i,::])
    F = np.exp(-dist_obs*observation_loc)
    for i in range(p):
        F[i,::] = F[i,::] / np.sum(F[i,::])
    m = np.array([prior_guess]*N)
    W = np.exp(-dist_state*movement_cor) * movement_var
    V = np.identity(p) * observation_noise
    C = np.identity(N) * prior_var
    # Create and return model
    model = kf.DLM([G, W], [F, V], [m, C])
    return model


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
