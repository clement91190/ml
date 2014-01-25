import math
import random
import os
import sys
import time
import cPickle
import numpy
import theano
import theano.tensor as T

rng = numpy.random.RandomState(1234)

class MLP(object):
    def __init__(self, input, n_in, n_hidden, batch_size, n_out):
        self.hiddenLayer = HiddenLayer(input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       batch_size=batch_size,
                                       activation=T.tanh)

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            batch_size=batch_size,
            n_out=n_out)

#norm L1, L2
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        self.rmse = self.logRegressionLayer.rmse
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.rmse

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, batch_size, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W1', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out, 1), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b1', borrow=True)

        self.W = W
        self.ones = theano.shared(value=numpy.ones((1, batch_size),
                                                 dtype=theano.config.floatX),
                               name='b1', borrow=True)
        self.b = b

        print self.W.get_value().shape
        print self.b.get_value().shape
        print self.ones.get_value().shape
        lin_output = T.dot(input, self.W) + T.transpose(T.dot(self.b, self.ones))
        print "hidden"
        print lin_output
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
        
        
class LogisticRegression():
    def __init__(self, input, n_in, n_out, batch_size):
        """ input is a matrix containing a minibatch of vector """
        self.input = input

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.ones = theano.shared(value=numpy.ones((1, batch_size),
                                                 dtype=theano.config.floatX),
                               name='ones', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out, 1),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        # compute vector of class-membership probabilities in symbolic form

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + T.transpose(T.dot(self.b, self.ones)))
        self.params = [self.W, self.b]  

    def rmse(self, proba):
        """ rmse over the minibatch """
        print "coucou"
        #self.result()
        return T.sqrt(T.mean(T.sqr(proba - self.p_y_given_x)))

    def result(self):
        print " W"
        print self.W.get_value()
        print " b"
        print self.b.get_value()
        print "input"
        print self.input

def learn():
    
    #hyper-parameters : 
    learning_rate = 3.0
    batch_size = 10    
    n_hidden = 500    
    L1_reg =0.00
    L2_reg =0.0001
    n_epochs = 100000

    datasets, size = load_data()

    img_size =  "image de taille " , math.sqrt(size)
    print img_size
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]
    valid_set_x, valid_set_y = datasets[1]


    #print test_set_y.dtype
    # compute number of minibatches for training, validation and testing
    print "number of images in train :", train_set_x.shape[0]
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size
    
    print "number of batches in train, valid, test", n_train_batches, n_valid_batches, n_test_batches

    print "building the model"

    y = T.fmatrix('y')
    x = T.fmatrix('x')
   
    classifier = MLP(input=x, n_in=size, n_out=37, batch_size=batch_size, n_hidden=n_hidden)
    cost = classifier.rmse(y) \
        + L1_reg * classifier.L1 \
        + L2_reg * classifier.L2_sqr

    
    # compiling a Theano function that computes the mistakes that are made by
    validate_model = theano.function(
        inputs=[x, y],
        outputs=classifier.rmse(y))
    valid_func = lambda i: validate_model(
        valid_set_x[i * batch_size: (i + 1) * batch_size],
        valid_set_y[i * batch_size: (i + 1) * batch_size] )
    test_model = theano.function(
        inputs=[x, y],
        outputs=classifier.rmse(y))
    test_func = lambda i: test_model(
        test_set_x[i * batch_size: (i + 1) * batch_size],
        test_set_y[i * batch_size: (i + 1) * batch_size])

    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    for param, gparam in zip(classifier.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    train_model = theano.function(
        inputs=[x, y],
        updates=updates,
        outputs=classifier.rmse(y))

    print '... training the model'
    # early-stopping parameters
    patience = 500000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
   
  #givens={
  #          x: test_set_x[index * batch_size: (index + 1) * batch_size],
  #          y: test_set_y[index * batch_size: (index + 1) * batch_size]}  # the model on a minibatch
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            x1 = train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y1 = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]  # the model on a minibatch
            
            train_model(x1, y1)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
           
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [valid_func(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_func(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

def load_data():
    print '... loading data'
    dataset = 'data/training_set_v1/training_small.pkl'
    # Load the dataset
    with open(dataset, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    print "... done "   
    return ((train_set, valid_set, test_set), train_set[0].shape[1])

def main():
    learn()

if __name__ == "__main__":
    main()
