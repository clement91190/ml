# -*- coding: utf-8 -*- 
import numpy as np
#Implementation of an interface to different optimization algorithms.


class OptimizationAlgorithm():
    """ """
    def __init__(self, fitness):
        self.fitness = fitness

    def initialize(self, params):
        """ params are initialization parameters for the algorithm """
        pass
    
    def optimize(self, params, verbose):
        """
        params are hyper parameter of the optimization algorithm
        verbose is to write results during the optimization process
        """
        pass
    
    def get_result(self):
        """ return the best value returned by the algorithm """


class Test():
    def __init__(self, optim_algorithm, init_param, optim_param):
        self.fitness = rosenbrock_function
        self.optim_algorithm = optim_algorithm
        self.init_param = init_param
        self.optim_param = optim_param

    def optimize(self):
        self.optim_algorithm.initialize(self.init_param)
        self.optim_algorithm.optimize(self.optim_param, True)
        print self.optim_algorithm.get_result()

def rosenbrock_function(x):
    """ x is a numpy vector """
    res = 0
    for i in range(x.shape[0])[:-1]:
        res += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1.0) ** 2
    return res


def square_function(x):
    return np.sum(x * x)
