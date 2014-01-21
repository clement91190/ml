# -*- coding: utf-8 -*-
#implementation of simulated annealing
from interface import OptimizationAlgorithm, Test, rosenbrock_function
import random
import numpy as np
import math


class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(self, fitness):
        OptimizationAlgorithm.__init__(self, fitness)

    def initialize(self, x0):
        self.verbose = True
        self.x = self.x_best = x0
        self.e = self.e_best = self.energy(x0)
        self.k = 1
        self.t = 10.0
        self.size = x0.shape[0]

    def energy(self, x):
        e = self.fitness(x)
        if self.verbose:
            print "fitness -> ", e
        return e

    def update_temperature(self):
        """ update the temperature """
        cooling = 0.9
        self.t *= cooling
        print "temp", self.t

    def neighbour(self, x):
        """ choice of a random neigbour with gaussian probability
         mean -> x
         sigma -> id * t """
        return np.random.multivariate_normal(x, self.t * np.identity(self.size))

    def acceptance(self):
        if self.e_new < self.e:
            return 1.0
        else:
            #we might still accept a solution
            try:
                assert(self.e_new - self.e < 0 )
                prob  = math.exp((self.e_new - self.e) / self.t)
                print "prob", prob
                return prob
            except:
                return 0.0

    def optim_step(self):
        self.update_temperature()
        self.x_new = self.neighbour(self.x)
        self.e_new = self.energy(self.x_new)
        if self.acceptance() > 0.5:
            self.x = self.x_new
            print "accept !"
            self.e = self.e_new
        if self.e_new < self.e_best:
            print "new best ->", self.e_new
            self.x_best = self.x_new
            self.e_best = self.e_new
        self.k += 1

    def optimize(self, kmax, verbose):
        self.verbose = verbose
        self.kmax = kmax
        while self.k < self.kmax:
            self.optim_step()
        return self.x_best

    def get_result(self):
        return (self.x_best, self.e_best)


def main():
    algo = SimulatedAnnealing(rosenbrock_function)
    t = Test(algo, 2 * np.ones(5), 1000)
    t.optimize()

if __name__ == "__main__":
    main()
