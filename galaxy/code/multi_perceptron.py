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
        n_out = architecture[-1]
        n_hiddens = architecture[:-2]
        n_hidden_layers = len(n_hiddens)
        if params is None:
            Wl, bl = None, None
            hidden_params = []
            for n_h in range(n_hidden_layers):
                hidden_params += [None, None]
        else:
            Wl, bl = params[-2:]
            hidden_params = params[:-2]

        #building hidden layers 
        self.hiddenLayers = []
        input_temp = input
        for i, n_in_temp in enumerate(n_hiddens):
            n_hidden_temp = architecture[i + 1]
            Wh, bh = hidden_params[2*i: 2*(i+1)]
            self.hiddenLayers.append(
                HiddenLayer(
                    input=input_temp,
                    n_in=n_in_temp, n_out=n_hidden_temp,
                    W=Wh, b=bh))
            input_temp = self.hiddenLayers[-1].output
        
        #building last layer
        self.logRegressionLayer = LogisticRegression(
            x=input_temp,
            y=y,
            n_in=architecture[-2],
            n_out=n_out,
            W=Wl, b=bl)
        
        L1_sum = lambda W:abs(W).sum()
        L2_sum = lambda W:abs(W ** 2).sum()

        sum_W_hidden1 = reduce(lambda W1, W2: L1_sum(W1) + L1_sum(W2), [hidden.W for hidden in self.hiddenLayers]) 
        sum_W_hidden2 = reduce(lambda W1, W2: L2_sum(W1) + L2_sum(W2), [hidden.W for hidden in self.hiddenLayers]) 

        self.L1 = sum_W_hidden1 + abs(self.logRegressionLayer.W).sum()

        self.L2_sqr = sum_W_hidden2 + (self.logRegressionLayer.W ** 2).sum()

        self.cross_err = self.logRegressionLayer.cross_err + self.L1_reg * self.L1 + self.L2_reg * self.L2_sqr

        self.least_square = self.logRegressionLayer.least_square
        
        self.proba = self.logRegressionLayer.p_1

        self.params = reduce(lambda x, y: x + y, [hidden.params for hidden in self.hiddenLayers]) + self.logRegressionLayer.params
        self.architecture = architecture

