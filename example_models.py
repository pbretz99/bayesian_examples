import numpy as np

import kalman_filter as kf
import temp_grid as tg
import utilities as ut

def default_RWWN_model():
    # Create default model
    # Random walk with noise
    V, W = 0.5, 0.1
    obs_params = [np.array([1]), np.array([[V]])]
    state_params = [np.array([1]), np.array([[W]])]
    prior = [np.array([80]), np.array([[1]])]
    source = np.array([0])
    model = kf.DLM(state_params, obs_params, prior, source)
    return model

def default_RWWN_vel_model():
    # Create default model
    # Random walk with noise and time-varying velocity
    V, W_x, W_v = 10, 0.1, 0.5
    prior_x, prior_v = 80, 1
    prior_x_var, prior_v_var = 1, 1
    obs_params = [np.array([[1, 0]]), np.array([[V]])]
    state_params = [np.array([[1, 1], [0, 1]]), np.array([[W_x, 0], [0, W_v]])]
    prior = [np.array([prior_x, prior_v]), np.array([[prior_x_var, 0], [0, prior_v_var]])]
    model = kf.DLM(state_params, obs_params, prior, np.array([0, 0]))
    return model 

def default_Heat_grid():
    # Create default model:
    # 5 meter x 5 meter room
    # State grid with approximately 1 meter spacing
    # Irregularly spaced thermometers
    # Heat source on the left and heat sink on the right
    # Prior guess of uniform 70 degrees
    # All variance (movement, observation, and prior) = 1
    dx, side_length = 1, 5
    obs_points = [[0.5, 0.9], [1.0, 4.0], [2.1, 2.3], [4.8, 3.2]]
    source_points = [[0.1, 2.5], [4.9, 2.5]]
    source_weights = [1, -1]
    state_grid, obs_grid = ut.make_grids(dx, side_length, np.array(obs_points))
    state_grid, source_grid = ut.make_grids(dx, side_length, np.array(source_points))
    source_terms = [source_grid, np.array(source_weights)]
    return state_grid, obs_grid, source_terms
    
def default_Heat_params():
    # Prior guess of uniform 70 degrees
    # All variance (movement, observation, and prior) = 1
    obs_noise, move_noise = 1, 1
    prior_guess, prior_var = 70, 1
    params = [obs_noise, move_noise, prior_guess, prior_var]
    return params
