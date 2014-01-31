import cPickle
import os
import sys
import time
#from code.multi_perceptron import MLP
from code.conv_net import Lenet5

import numpy as np

import theano
import theano.tensor as T


class NNTester():
    def __init__(self, test_set):

        self.x = T.matrix("x")
        self.y = T.matrix("y")
        self.test_set = test_set
        self.batch_size, self.feature_size = self.test_set.shape

        params, architecture = self.load_model()
        #self.neural_net = MLP(self.x, self.y, architecture, params)
        self.neural_net = Lenet5(self.x, self.y, architecture, params)
        self.predict = theano.function(inputs=[self.x], outputs=self.neural_net.proba)

    def load_model(self, fich="model_nn.tkl"):
        with open(fich) as f:
            params, architecture = cPickle.load(f)
            return (params, architecture)

    def test(self):
        return self.predict(self.test_set)


class NNTrainer():
    def __init__(self, datasets, n_train_batches=1):
        """ datasets = (train_set, valid_set, test_set) """
        self.inputs, self.labels = datasets[0]  # inputs, labels from train_set
        self.datasets = datasets
        self.N, self.feature_size = self.inputs.shape
        self.batch_size = int(self.N / n_train_batches)
        self.n_train_batches = n_train_batches
        self.label_size = self.labels.shape[1]
        self.training_steps = 1000
        self.learning_rate = 1.0

        # Declare Theano symbolic variables (x -> inputs, y-> labels)
        self.x = T.matrix("x")
        self.y = T.matrix("y")
 
        #for first time only
        nkerns = [20, 30]
        mlp_in = nkerns[1] * 6 * 6
        mlp_architecture = [mlp_in, 100, 100, self.label_size]
        architecture = (self.feature_size, nkerns, mlp_architecture)
        params = None
       
        print "... load model"
        #params, architecture = self.load_model()
        print architecture

        self.neural_net = Lenet5(self.x, self.y, self.batch_size, architecture, params)
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

        self.test = theano.function(inputs=[self.x, self.y],outputs=self.neural_net.least_square)
    
    def load_model(self, fich="model_nn.tkl"):
        print "loading previous model"
        with open(fich) as f:
            params, architecture = cPickle.load(f)
        return (params, architecture)

    def save_model(self):
        with open("model_nn.tkl", 'w') as fich:
            cPickle.dump(([p.get_value() for p in self.neural_net.params], self.neural_net.architecture), fich)

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
                    this_validation_loss = np.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                        (epoch, minibatch_index + 1, self.n_train_batches,
                        this_validation_loss * 100.))

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


def load_data():
    print '... loading data'
    dataset = 'data/training_set_v1/training.pkl'
    # Load the dataset
    with open(dataset, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    print "... done "   
    return (train_set, valid_set, test_set)


def main():
    train, validate, test = load_data()
   # N, feats, n_out = 100, 10, 2
   # train = (rng.randn(N, feats), 0.5 * np.ones((N, n_out)))
   # validate = (rng.randn(N, feats), 0.5 * np.ones((N, n_out)))
   # test = (rng.randn(N, feats), 0.5 * np.ones((N, n_out)))
    D = (train, validate, test)
    trainer = NNTrainer(D, n_train_batches=200)
    trainer.train_loop()
    #print trainer.predict(D[0])
    print trainer.neural_net.params

if __name__ == "__main__":
    main()

