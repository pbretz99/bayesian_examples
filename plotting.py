# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:46:28 2021

@title: Plotting
@author: Philip Bretz
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import beta

import utilities as ut

def beta_plot(a, b, text, ax=None, **kwargs):
    ax = ax or plt.gca()
    x = np.linspace(0.01, 0.99, 100)
    ax.plot(x, beta.pdf(x, a, b), **kwargs)
    ax.fill_between(x, beta.pdf(x, a, b), alpha=0.2)
    title = text + ": a = %.1f,  b = %.1f" % (a, b)
    ax.title.set_text(title)
    
def one_d_kalman_plot(theta, Y, fit_values, fit_var):
    fig, ax = plt.subplots(figsize=(10, 5))
    df = ut.one_d_kalman_df(theta, Y, fit_values, fit_var)
    colors = ['slategray', 'forestgreen', 'darkblue']
    styles = ['-', '--', '-']
    df[['State', 'Observations', 'Filter']].plot(ax=ax, color=colors, style=styles)
    ax.fill_between(df.index, df['Lower'].astype(float), df['Upper'].astype(float), alpha=0.25)
    plt.title('Kalman Filter (90% CI)')
    plt.show()

def target_plot(Y, fit_values, pad=1):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(np.min(Y[::,0])-pad, np.max(Y[::,0])+pad)
    ax.set_ylim(np.min(Y[::,1])-pad, np.max(Y[::,1])+pad)
    ax.plot(Y[::,0], Y[::,1], '-o', alpha=0.5)
    ax.plot(fit_values[::,0], fit_values[::,1], c='green', alpha=0.75)
    plt.title('Target Tracking')
    plt.show()
