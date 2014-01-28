# -*- coding: utf-8 -*- 
import numpy as np
#Implementation of the Nelder-Mead downhill method (or simplex) 
import tools
from interface import OptimizationAlgorithm, Test, square_function, rosenbrock_function
import matplotlib.pyplot as plt


class Simplex(OptimizationAlgorithm):
    """ """
    def __init__(self, fitness):
        self.fitness = fitness

    def initialize(self, (points, size)):

        self.size = size
        self.popul_size = size + 1
        if points is None:
            self.points = tools.random_init(self.popul_size, self.size)
        else:
            self.points = points
        self.verbose = True
        self.evaluations = [self.fitness(p) for p in self.points]
        self.order = []
        self.stored_results = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def evaluation_of_new(self):
        """ evaluation of the elements that require an evaluation"""
        for i in [elem for elem in range(self.popul_size) if self.evaluations[elem] is None]:
            val = self.fitness(self.points[i])
            self.evaluations[i] = val

    def restart(self):
        print "restart"
        x1 = self.points[self.order[0]]
        dist = 0
        for p1 in self.points:
            for p2 in self.points:
                new_d = np.linalg.norm(p1 - p2)
                if new_d > dist:
                    dist = new_d
        self.points = tools.random_init(self.popul_size, self.size, x1, 0.5 * dist * np.identity(self.size))
        self.evaluations = [self.fitness(p) for p in self.points]

    def homotethia(self):
        """ homotethia of scale 0.5 centered on best point """
        print "homotethia"
        x1 = self.points[self.order[0]]
        for p in self.order[1:]:
            xi = self.points[p]
            self.points[p] = 0.5 * (xi + x1)
            self.evaluations[p] = None

    def sort_points(self):
        self.order = sorted(range(self.popul_size), key=lambda i: self.evaluations[i])

    def get_center_of_mass(self):
        self.x0 = np.array(self.points).mean(axis=0)  # center of gravity

    def get_reflection_point(self):
        "xr is the reflection points in comparaison to x0"
        self.xr = 2.0 * self.x0 - self.points[self.order[-1]]
        self.eval_r = self.fitness(self.xr)

    def get_extension_point(self):
        "xr is the reflection points in comparaison to x0"
        self.xe = 3.0 * self.x0 - 2.0 * self.points[self.order[-1]]
        self.eval_e = self.fitness(self.xe)

    def get_contraction_point(self):
        "xr is the reflection points in comparaison to x0"
        self.xc = 0.5 * self.x0 + 0.5 * self.points[self.order[-1]]
        self.eval_c = self.fitness(self.xc)

    def optimize_step(self):
        self.sort_points()
        self.stored_results.append([self.get_result()[1]])
        self.get_center_of_mass()
        self.get_reflection_point()
        if self.eval_r < self.evaluations[self.order[-2]]:
            self.get_extension_point()
            if self.eval_e < self.eval_r:
                print "extension"
                self.points[self.order[-1]] = self.xe
                self.evaluations[self.order[-1]] = self.eval_e
            else:
                print "keep reflexion"
                self.points[self.order[-1]] = self.xr
                self.evaluations[self.order[-1]] = self.eval_r

        else:
            self.get_contraction_point()
            if self.eval_c < self.evaluations[self.order[-2]]:
                print "contraction"
                self.points[self.order[-1]] = self.xc
                self.evaluations[self.order[-1]] = self.eval_c
            else:
                self.homotethia() 
        self.evaluation_of_new()

    def optimize(self, kmax, verbose):
        for k in range(kmax):
            self.optimize_step()
            if k % 20 == 0:
                self.restart()
            self.plot()
            raw_input()

    def get_result(self):
        """ return the best value returned by the algorithm """
        return (self.points[self.order[0]], self.evaluations[self.order[0]])

    def plot(self):
        """ plot for 2d case """
        x = [p[0] for p in self.points]
        y = [p[1] for p in self.points]
        x.append(-1)
        y.append(-1)
        x.append(1)
        y.append(1)
        self.ax.plot(x, y ,'ro')
        self.fig.show()
        raw_input()
        self.ax.clear()


def main():
    algo = Simplex(square_function)
    #algo = Simplex(rosenbrock_function)
    t = Test(algo, (None, 2), 1000)
    t.optimize()
    x = []
    y = []
    for i, gen in enumerate(algo.stored_results):
        for p in gen:
            x.append(i)
            y.append(min(p, 5.0))
    plt.plot(x, y)
    plt.show()
if __name__ == "__main__":
    main()
