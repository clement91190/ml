import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import math
from multi_perceptron import MLP


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
        rng = np.random.RandomState(1234)
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))

        if W is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX)
            b = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        
        self.W = theano.shared(W)
        self.b = theano.shared(b)

        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]


class Lenet5():
    def __init__(self, input, y, architecture, params=None):
        #TODO record architecture
        if params is None:
            self.mlp_params = None
            W0, b0 = None, None
            W1, b1 = None, None
        else:
            W0, b0 = params[0:2]
            W1, b1 = params[2:4]
            self.mlp_params = params[4:]

        self.n_in, self.nkerns, self.mlp_arch = architecture
        self.image_width = int(math.sqrt(self.n_in))
        print self.image_width
    
        #TODO find a way of reshaping without giving the batch size !

        self.layer0_input = T.reshape(input, (-1, 1, self.image_width, self.image_width))

        self.layer0 = LeNetConvPoolLayer(input=self.layer0_input,
                image_shape=(None, 1, self.image_width, self.image_width),
                filter_shape=(self.nkerns[0], 1, 9, 9), poolsize=(2, 2),
                W=W0, b=b0)

        self.layer1 = LeNetConvPoolLayer(input=self.layer0.output,
                image_shape=(None, self.nkerns[0], 14, 14),
                filter_shape=(self.nkerns[1], self.nkerns[0], 7, 7), poolsize=(2, 2),
                W=W1, b=b1)

        self.mlp_input = self.layer1.output.flatten(2)
        #mlp_in = self.nkerns[1] * 4 * 4
        self.mlp = MLP(input=self.mlp_input, y=y, architecture=self.mlp_arch, params=self.mlp_params)

        self.cross_err = self.mlp.cross_err

        self.least_square = self.mlp.least_square
        
        self.proba = self.mlp.proba

        self.params = self.layer0.params + self.layer1.params + self.mlp.params
        self.architecture = architecture

