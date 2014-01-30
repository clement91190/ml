from code.logistic_reg import LogisticRegression

import numpy

import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        rng = numpy.random.RandomState(1234)
        self.input = input
        if W is None:
            W = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W *= 4

            #W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b = numpy.zeros((1, n_out), dtype=theano.config.floatX)
            #b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = theano.shared(W)
        self.b = theano.shared(b)

        self.output = activation(T.dot(input, self.W) + T.dot(T.ones_like(T.eye(input.shape[0], 1)), self.b))
        # parameters of the model
        self.params = [self.W, self.b]


class MLP(object):
    def __init__(self, input, y, architecture, params=None):
#TODO change this for any architecture.
        self.L1_reg = 0.00
        self.L2_reg = 0.0001
        n_in, n_hidden, n_out = architecture
        if params is None:
            Wh, bh, Wl, bl = None, None, None, None
        else:
            Wh, bh, Wl, bl = params
        self.hiddenLayer = HiddenLayer(input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       W=Wh, b=bh)

        self.logRegressionLayer = LogisticRegression(
            x=self.hiddenLayer.output,
            y=y,
            n_in=n_hidden,
            n_out=n_out,
            W=Wl, b=bl)

        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()

        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        self.cross_err = self.logRegressionLayer.cross_err + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        self.least_square = self.logRegressionLayer.least_square
        
        self.proba = self.logRegressionLayer.p_1

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.architecture = architecture

