# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:55:14 2021

@title: Bayesian Coin Flip
@author: Philip Bretz
"""

'''
With a prior distibution of Beta(a,b),
the posterior distribution from k heads after
n flips is Beta(a',b') where a' = a + k and
b' = b + n - k.

Given those results, the probability of a head 
follows a Beta-Binomial distribution. This simplifies 
to B(a'+1,b')/B(a',b'), where B(x,y) is the beta 
function (different than the beta distribution).
'''

import matplotlib.pyplot as plt

from scipy.special import beta as B

import plotting as pl

active = True
while active:
    a = input('Input prior hyperparameter alpha: ')
    b = input('Input prior hyperparameter beta: ')
    flips = input('Input number of coin flips: ')
    heads = input('Input number of heads: ')
    a_new = float(a)+int(heads)
    b_new = float(b)+int(flips)-int(heads)
    if a_new == 0.0:
        bayes_prob = 0.0
        print('\nNote: improper posterior distribution!')
    elif b_new == 0.0:
        bayes_prob = 1.0
        print('\nNote: improper posterior distribution!')
    else:
        bayes_prob = B(a_new+1,b_new)/B(a_new,b_new)
        if float(a) != 0.0 and float(b) != 0.0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            pl.beta_plot(float(a), float(b), 'Prior', ax1)
            pl.beta_plot(a_new, b_new, 'Posterior', ax2)
            plt.show()
        else:
            print('\nNote: improper prior distribution!')
            fig, ax = plt.subplots(1, figsize=(5, 5))
            pl.beta_plot(a_new, b_new, 'Posterior', ax)
            plt.show()
    freq_prob = int(heads)/int(flips)
    print('\nBayesian probability of heads: ', round(bayes_prob, 3))
    print('Frequentist MLE probability of heads: ', round(freq_prob, 3))
    again = input('Run again? Y/N: ')
    if again != 'Y':
        break