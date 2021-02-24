# bayesian_examples
Contains a few different examples of Bayesian methods, with varying complexity.

## Required Libraries: 
matplotlib, numpy, pandas, scipy

## Example 1: Coin Flip (run Coin_Flip.py)
The program prompts you to choose beta hyperparameters for the prior distribution on the probability of flipping heads, the number of heads, and the number of flips. It then plots the prior and posterior distributions (if they are proper) and compares the Bayesian estimate to the frequentist estimate.

## Example 2: Random Walk with Noise (run RWWN.py)
This program simulates a random walk with noise, and then applies the Kalman filter to the simulated data to estimate the true state. It then plots the true state, observations, filtered estimates, and 90% credible interval. You can use default parameters or choose your own.

## Example 3: Random Walk with Velocity and Noise (run RWWN_vel.py)
This program simulates a random walk with time-varying velocity and observational noise, and then applies the Kalman filter to the simulated data to estimate the true state (position and velocity). It then plots the true position, observed position, estimated position, and 90% credible intervals. You can use the default parameters or choose your own.

## Example 4: Multi Dimensional Kalman Filter (run Heat_Eqn.py)
This program simulates temperature on a grid, with irregularly spaced thermometers. It then applies the Kalman filter to the simulated data to estimate the temperature at each grid point.

## Example 5: Target Tracking (run Target_Tracking.py)
This program simulates the movement of a target in 2-d, with noisy observations. The target's movement is governed by a random walk with time-varying velocity that follows an Integrated Ornstein-Uhlenbeck process (essentially, the speed moves randomly, but tends towards a pre-set value). It then applies the Kalman filter to the simulated 'pings' to estimate the true position of the target at each time.
