import cPickle
import os
import sys
import time
from code.nn.multi_perceptron import MLP
#from code.nn.conv_net import Lenet5

import numpy as np

import theano
import theano.tensor as T


class NNTrainer():
    def __init__(self, datasets, method, learn_type, load=True, learning_rate=0.3, n_train_batches=1):
        """ datasets = (train_set, valid_set, test_set) """

        self.learn_type = learn_type
        self.method = method
        self.inputs, self.labels = datasets[0]  # inputs, labels from train_set
        self.datasets = datasets
        self.N, self.feature_size = self.inputs.shape
        self.batch_size = int(self.N / n_train_batches)

        print "batch size : ", self.batch_size

        self.n_train_batches = n_train_batches
        self.label_size = self.labels.shape[1]
        self.training_steps = 1000
        self.learning_rate = learning_rate

        # Declare Theano symbolic variables (x -> inputs, y-> labels)
        self.x = T.matrix("x")
        self.y = T.matrix("y")

        #for first time only
        #nkerns = [20, 30]
        #mlp_in = nkerns[1] * 4 * 4
        #mlp_architecture = [mlp_in, 100, 100, self.label_size]
        #architecture = (self.feature_size, nkerns, mlp_architecture)
        architecture = [self.feature_size, 150, 100, 80, self.label_size]
        params = None

        if load:
            print "... load model"
            params, architecture = self.load_model()
        print architecture

        #self.feature_size, nkerns, mlp_architecture = architecture
        self.neural_net = MLP(self.x, self.y, architecture, params)
        #self.neural_net = Lenet5(self.x, self.y, architecture, params)
        self.cross_error = theano.function(inputs=[self.x, self.y], outputs=self.neural_net.cross_err)

        gparams = []
        for param in self.neural_net.params:
            gparam = T.grad(self.neural_net.cross_err, param)
            gparams.append(gparam)

        self.updates = []
        for param, gparam in zip(self.neural_net.params, gparams):
            self.updates.append((param, param - self.learning_rate * gparam))

        self.train = theano.function(
            inputs=[self.x, self.y],
            outputs=self.neural_net.least_square,
            updates=self.updates)

        self.predict = theano.function(inputs=[self.x], outputs=self.neural_net.proba)

        self.test = theano.function(inputs=[self.x, self.y], outputs=self.neural_net.least_square)
        #self.simplifier = Simplifier()

    def load_model(self):
        print "loading previous model"
        fich = 'data/' + self.method + '/' + self.learn_type + '/model.tkl'
        with open(fich) as f:
            params, architecture = cPickle.load(f)
        return (params, architecture)

    def save_model(self):
        fich = 'data/' + self.method + '/' + self.learn_type + '/model.tkl'
        with open(fich, 'w') as fich:
            cPickle.dump(([p.get_value() for p in self.neural_net.params], self.neural_net.architecture), fich)

    def real_error(self, batch_x, batch_label):
        err = self.test(batch_x, batch_label)
        batch_y = self.predict(batch_x)
        batch_modif = self.simplifier.modify_batch(batch_y)
        sq = np.square(batch_modif - batch_label)
        print "difference :"
        res = np.sqrt(sq.mean())
        print res - err
        return res

    def train_loop(self):

        ###############
        # TRAIN MODEL #
        ###############
        print '... training the model'
        # early-stopping parameters
        patience = 50000  # look as this many examples regardless
        n_epochs = 10000
        patience_increase = 2  # wait this much longer when a new best is
                                    # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                    # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                    # go through this many
                                    # minibatche before checking the network
                                    # on the validation set; in this case we
                                    # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            #print [p.get_value() for p in self.neural_net.params]
            for minibatch_index in xrange(self.n_train_batches):
                self.train(
                    self.inputs[minibatch_index * self.batch_size: (minibatch_index + 1) * self.batch_size],
                    self.labels[minibatch_index * self.batch_size: (minibatch_index + 1) * self.batch_size])
                # iteration number
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self.test(self.datasets[1][0], self.datasets[1][1])]
                    train_losses = [self.test(self.inputs[0:self.batch_size], self.labels[0:self.batch_size])]
                    this_validation_loss = np.mean(validation_losses)
                    this_train_loss = np.mean(train_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f, train_error %f %% ' % \
                        (epoch, minibatch_index + 1, self.n_train_batches,
                        this_validation_loss * 100., this_train_loss * 100.))
                    # if we got the best validation score until now

                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                        improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [self.test(self.datasets[2][0], self.datasets[2][1])]
                        test_score = np.mean(test_losses)
                        #test_score = self.real_error(self.datasets[2][0], self.datasets[2][1])

                        print(('     epoch %i, minibatch %i/%i, test error of best'
                        ' model %f %%') %
                            (epoch, minibatch_index + 1, self.n_train_batches,
                            test_score * 100.))
                        self.save_model()

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

    def predict(self, test_inputs):
#TODO add confidence ? 
        return self.predict(test_inputs)
