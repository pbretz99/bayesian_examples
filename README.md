# bayesian_examples
Contains a few different examples of Bayesian methods, with varying complexity.

## Required Libraries: 
matplotlib, numpy, pandas, scipy

## Example 1: Coin Flip (run coin_flip.py)
The program prompts you to choose beta hyperparameters for the prior distribution on the probability of flipping heads, the number of heads, and the number of flips. It then plots the prior and posterior distributions (if they are proper) and compares the Bayesian estimate to the frequentist estimate.

## Example 2: Random Walk with Noise (run 1d_kalman.py)
This program simulates a random walk with noise, and then applies the Kalman filter to the simulated data to estimate the true state. It then plots the true state, observations, filtered estimates, and 90% credible interval. You can use default parameters or choose your own.

## Example 3: Random Walk with Velocity and Noise (not written yet)
This program simulates a random walk with time-varying velocity and observational noise, and then applies the Kalman filter to the simulated data to estimate the true state (position and velocity). It then plots the true position, observed position, estimated position, and 90% credible intervals. You can use the default parameters or choose your own. I have not written this one yet.

## Example 4: Multi Dimensional Kalman Filter (not finished)
This program simulates temperature on a grid, with irregularly spaced thermometers. It then applies the Kalman filter to the simulated data to estimate the temperature at each grid point. This one is not up and running yet.
