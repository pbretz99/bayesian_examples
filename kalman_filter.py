# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:19:04 2021

@title: Kalman Filter mk. 2
@author: Philip Bretz
"""

import numpy as np

from scipy.stats import multivariate_normal
from scipy.stats import norm

# Note: all parameters take the form [m, sd]
class DLM:
    def __init__(self, state_params, space_params, state_prior, source):
        self.state_params = state_params
        self.space_params = space_params
        self.filter_params = state_prior
        self.source = source #source is same length as state
    
    def update(self, y):
        state, space = self.state_params, self.space_params
        G, G_trans, W = state[0], state[0].transpose(), state[1]
        F, F_trans, V = space[0], space[0].transpose(), space[1]
        # One-step ahead state
        a = G.dot(self.filter_params[0])+self.source
        R = G.dot(self.filter_params[1].dot(G_trans))+W
        # One-step ahead predictive
        f = F.dot(a)
        Q = F.dot(R.dot(F_trans))+V
        # Filtered parameters
        Q_inv = np.linalg.inv(Q)
        m = a+R.dot(F_trans.dot(Q_inv.dot(y-f)))
        C = R-R.dot(F_trans.dot(Q_inv.dot(F.dot(R))))
        self.filter_params = [m, C]
        return [m, C]
    
    def kfilter(self, Y):
        fit, var = [], []
        for y in Y:
            current_params = self.update(y)
            fit.append(current_params[0])
            var.append(current_params[1])
        return np.array(fit), np.array(var)
    
    def N_step_ahead(self, N):
        state = self.state_params
        G, G_trans, W = state[0], state[0].transpose(), state[1]
        [m, C] = self.filter_params
        for i in range(N):
            m_new = G.dot(m)+self.source
            C_new = G.dot(C.dot(G_trans))+W
            m, C = m_new, C_new
        return [m, C]
    
    def simulate(self, length, init):
        state, space = self.state_params, self.space_params
        G, W = state[0], state[1]
        F, V = space[0], space[1]
        if init.size == 1:
            # Single dimensional
            theta = np.zeros((length, 1))
            Y = np.zeros((length, 1))
            nu = norm.rvs(0, V, length)
            omega = norm.rvs(0, W, length)
            theta[0,::] = init
            Y[0,::] = F * theta[0,::] + nu[0]
            for i in range(1, length):
                theta[i,::] = G * theta[i-1,::] + self.source + omega[i]
                Y[i,::] = F * theta[i,::] + nu[i]
        else:
            # Multi dimensional
            N, p = int(np.sqrt(W.size)), int(np.sqrt(V.size))
            theta = np.zeros((length, N)) # Each row is a given time
            Y = np.zeros((length, p)) # Ditto
            nu = multivariate_normal.rvs(np.zeros(p), V, length).reshape((length, p))
            omega = multivariate_normal.rvs(np.zeros(N), W, length)
            theta[0,::] = init
            Y[0,::] = F.dot(theta[0,::]) + nu[0,::]
            for i in range(1, length):
                theta[i,::] = G.dot(theta[i-1,::]) + self.source + omega[i,::]
                Y[i,::] = F.dot(theta[i,::]) + nu[i,::]
        return theta, Y
