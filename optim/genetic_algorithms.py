# -*- coding: utf-8 -*-
#implementation of simulated annealing
from interface import OptimizationAlgorithm, Test, rosenbrock_function
import random
import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm(OptimizationAlgorithm):
    """ implementation of Genetic Algorithm
        scale the parameter to be in [0, 1] intervals
    """
    def __init__(self, fitness):
        OptimizationAlgorithm.__init__(self, fitness)

    def initialize(self, (population, popul_size, size)):
        """param : - population a prepopulated set, otherwise use size to generate
        them randomly
                - popul_size is the size of the population
                - size is the size of each vector
        """
        self.verbose = True
        self.size = size
        self.popul_size = popul_size
        assert(self.popul_size % 4 == 0)
        if population is None:
            self.random_init()
        else:
            self.population = population
        self.evaluations = [self.fitness(p) for p in self.population]
        self.killed = []
        self.stored_results = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def random_init(self):
        """ generate the population with random gaussian distribution """
        self.population = []
        for i in range(self.popul_size):
            elem = np.random.multivariate_normal(
                0.5 * np.ones(self.size),
                np.identity(self.size))
            self.population.append(elem)

    def get_children(self, elem1, elem2):
        """ create the children of two elements """
        alpha = 0.9 * random.random() + 0.5
        father = self.population[elem1]
        mother = self.population[elem2]
        val1 = alpha * father + (1.0 - alpha) * mother
        val2 = (1.0 * alpha) * father - alpha * mother
        return (val1, val2)

    def cross_over(self):
        """ barycentric cros_over """
        i = 0
        if self.verbose:
            print "cross_over"
        while i < self.popul_size / 2:
            elem1 = self.get_random_elem()
            elem2 = self.get_random_elem()
            (val1, val2) = self.get_children(elem1, elem2)
            c1 = self.killed.pop()
            c2 = self.killed.pop()
            self.population[c1] = val1
            self.evaluations[c1] = None
            self.population[c2] = val2
            self.evaluations[c2] = None
            i += 2

    def evaluation_of_childrens(self):
        """ evaluation of children"""
        for i in [elem for elem in range(self.popul_size) if self.evaluations[elem] is None]:
            val = self.fitness(self.population[i])
            self.evaluations[i] = val

    def mutation(self):
        """
        we use a gaussian distribution with the covariance matrix,
        which is the covariance of the population
        """
        mutation_factor = 0.03
        for elem in range(self.popul_size):
            if random.random() < mutation_factor:
                cov_mat = np.cov(np.transpose(np.array(self.population)))
                self.population[elem] = np.random.multivariate_normal(
                    self.population[elem],
                    cov_mat)
                self.evaluations[elem] = None

    def get_random_elem(self):
        """ return an elem , for which we do have an evaluation """
        i = random.randint(0, self.popul_size - 1)
        while self.population[i] is None:
            i = random.randint(0, self.popul_size - 1)
        return i

    def kill(self, i):
        """ kill element i"""
        self.population[i] = None
        self.killed.append(i)

    def selection(self):
        """ half  tournament selection,
            half worse extermination"""
        if self.verbose:
            print "selection"
        
        worse = sorted(range(self.popul_size), key=lambda i: -self.evaluations[i])
        for i in worse[:self.popul_size / 2]:
            self.kill(i)
        """
        i = 0
        while i < self.popul_size / 4:
            i += 1
            elem1 = self.get_random_elem()
            elem2 = self.get_random_elem()
            if self.evaluations[elem1] > self.evaluations[elem2]:
                self.kill(elem1)
            else:
                self.kill(elem2)"""

    def optimize_step(self):
        """ classicla 4 steps of GAs """
        self.selection()
        self.cross_over()
        #self.mutation()
        self.evaluation_of_childrens()
        self.stored_results.append([self.best_of_population()[1]])
        assert(all([s is not None for s in self.population]))
        print "##std##", np.std(self.population)
        #self.plot()

    def optimize(self, k, verbose):
        """ repete for k step the optimize_step """
        for i in range(k):
            self.optimize_step()

    def best_of_population(self):
        """ return the best vector of the population """
        elem = min(range(self.popul_size), key=lambda elem: self.evaluations[elem])
        return (self.population[elem], self.evaluations[elem])

    def get_result(self):
        return self.best_of_population()
    
    def plot(self):
        """ plot for 2d case """
        x = [p[0] for p in self.population]
        y = [p[1] for p in self.population]
        x.append(-5)
        y.append(-5)
        x.append(10)
        y.append(10)
        self.ax.clear()
        self.ax.plot(x, y ,'ro')
        self.fig.show()
        raw_input()

def main():
    algo = GeneticAlgorithm(rosenbrock_function)
    t = Test(algo, (None, 1000, 4), 100)
    t.optimize()
    x = []
    y = []
    algo.ax.clear()
    for i, gen in enumerate(algo.stored_results):
        for p in gen:
            x.append(i)
            y.append(min(p, 5.0))
    plt.plot(x, y)
    plt.show()
if __name__ == "__main__":
    main()
