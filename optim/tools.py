# -*- coding: utf-8 -*-
""" file containing a list of useful tools in the
implementation of different optimization techniques """
import numpy as np


def random_init(popul_size, size, mu=None, sigma=None):
    """ generate the population with random gaussian distribution centerd on 0.5"""
    if mu is None:
        mu = 0.5 * np.ones(size)
    if sigma is None:
        sigma = np.identity(size)
    population = []
    for i in range(popul_size):
        elem = np.random.multivariate_normal(mu, sigma)
        population.append(elem)
    return population


