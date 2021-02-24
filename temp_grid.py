# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:24:01 2021

@title: Temperature Grid State-Space
@author: Philip Bretz
"""
import numpy as np

import kalman_filter as kf
import utilities as ut

def temp_model(state_grid, obs_grid, source_terms, params):
    source_grid, source_weights = source_terms[0], source_terms[1]
    # Unpack params: V, W, prior_guess, prior_var
    obs_noise, move_noise = params[0], params[1]
    prior_guess, prior_var = params[2], params[3]
    N, p, q = state_grid[0].size, obs_grid[0].size, source_grid[0].size
    dx = state_grid[0][1] - state_grid[0][0]
    # Define relevant parameters
    k, rho, c_p = 0.026, 1.2, 0.001 # The standard properties of air
    alpha = k / (rho * c_p)
    dt = 0.1*dx**2 / (2*alpha)
    lam, gamma = alpha * dt / (dx**2), dt / (rho * c_p)
    # Construct G, F, and Q
    G = heat_eq_state_mat(state_grid, dx, lam)
    F = heat_eq_obs_mat(obs_grid, state_grid, dx)
    Q = heat_eq_source_mat(source_terms, state_grid, gamma)
    # Construct W and V
    W = np.identity(N) * move_noise
    V = np.identity(p) * obs_noise
    # Construct m and C
    m = np.array([prior_guess]*N)
    C = np.identity(N) * prior_var
    # Create and return model
    model = kf.DLM([G, W], [F, V], [m, C], Q)
    return model

def heat_eq_state_mat(state_grid, dx, lam):
    N = state_grid[0].size
    tol = 0.01 * dx
    state_dist = ut.dist_mat(state_grid, state_grid)
    # Create matrix where the i-th row has a 1 at the j-th entry if the point
    # x_j is adjacent to x_i; i.e., an adjacency matrix
    M = np.abs(state_dist - dx) <= tol
    M = M.astype(int)
    for i in range(N):
        # Check for boundary points
        if np.sum(M[i,::]) < 4:
            compensate_ind = find_boundary(M[i,::], i, state_grid)
            # This compensation factor is a result of Neumann zero-flux boundary
            M[i,compensate_ind] = 2
    # This is the discretization of the heat equation
    # Each point is updated by a weighted sum
    # where its previous value has weight 1-4*lambda
    # and adjacent points have value lambda (or twice
    # that if they are opposite a boundary)
    G = (1-4*lam)*np.identity(N) + lam*M
    return G

def heat_eq_obs_mat(obs_grid, state_grid, dx):
    obs_dist = ut.dist_mat(obs_grid, state_grid)
    M = (obs_dist < dx).astype(int)
    row_sums = M.sum(axis=1)
    # Observations are an average of nearby (within dx)
    # state values
    F = M / row_sums[:, np.newaxis]
    return F

def heat_eq_source_mat(source_terms, state_grid, gamma):
    source_grid, source_weights = source_terms[0], source_terms[1]
    N, q = state_grid[0].size, source_grid[0].size
    source_dist = ut.dist_mat(source_grid, state_grid)
    Q = np.zeros(N)
    # Source vector is created by assigning a source to 
    # the closest state point with its corresponding source
    # weight, multiplied by the factor gamma
    for i in range(q):
        closest_state_ind = np.argmin(source_dist[i,::])
        Q[closest_state_ind] = source_weights[i]
    Q = gamma * Q
    return Q

def find_boundary(M_row, i, state_grid):
    # For the i-th row, returns j-th index if
    # x_i and x_j are adjacent and the point
    # opposite x_j is 'missing' (i.e., x_j is
    # on the boundary)  
    adj_ind = np.where(M_row == 1)[0]
    adj_points = np.zeros((adj_ind.size, 2))
    i = 0
    for ind in adj_ind:
        adj_points[i,0], adj_points[i,1] = state_grid[0][ind], state_grid[1][ind]
        i += 1
    missing_ind = []
    for i in range(adj_ind.size):
        p = adj_points[i,::]
        missing = True
        for q in adj_points:
            if q[0] == p[0] and q[1] != p[1]:
                missing = False
            elif q[0] != p[0] and q[1] == p[1]:
                missing = False
        if missing:
            missing_ind.append(adj_ind[i])
    return np.array(missing_ind)
