import theano.tensor as T
import cPickle
import pickle
import time
import os
import sys
import theano
import numpy as np


rng = np.random


class LogisticRegression():
    def __init__(self, x, y, n_in, n_out, W=None, b=None):
        if W is None:
            self.W = theano.shared(rng.random((n_in, n_out)))
            self.b = theano.shared(rng.random((1, n_out)))
        else:
            self.W = theano.shared(W)
            self.b = theano.shared(b)
        #self.n_batch = x.shape[0]
        activation = T.nnet.sigmoid
        #activation = T.tanh
        #self.p_1 = 1 / (1 + T.exp(-T.dot(x, self.W) - T.dot(np.ones((self.n_batch, 1)), self.b)))
        self.p_1 = activation(T.dot(x, self.W) + T.dot(T.ones_like(T.eye(x.shape[0], 1)), self.b))

        self.params = [self.W, self.b]
        self.def_cross_loss(y)
        self.def_least_square(y)
        self.params = [self.W, self.b]

    def reinit(self, x, y, W, b):
        self.W = theano.shared(W)
        self.b = theano.shared(b)
        #self.n_batch = x.shape[0]
        activation = T.nnet.sigmoid
        #self.p_1 = 1 / (1 + T.exp(-T.dot(x, self.W) - T.dot(np.ones((self.n_batch, 1)), self.b)))
        self.p_1 = activation(T.dot(x, self.W) + T.dot(T.ones_like(T.eye(x.shape[0], 1)), self.b))

        self.params = [self.W, self.b]
        self.def_cross_loss(y)
        self.def_least_square(y)
        self.params = [self.W, self.b]

    def def_cross_loss(self, y):
        self.cross = -y * T.log(self.p_1) - (1 - y) * T.log(1 - self.p_1)  # Cross-entropy loss function
        self.cross_err = self.cross.mean()

    def def_least_square(self, y):
        self.xent = T.sqr(y - self.p_1)
        self.least_square = T.sqrt(self.xent.mean())  # + 0.01 * (w ** 2).sum()# The cost to minimize


class LogisticRegressionTester():
    def __init__(self, test_set):

        self.x = T.matrix("x")
        self.y = T.matrix("y")
        self.test_set = test_set
        self.batch_size, self.feature_size = self.test_set.shape
        self.label_size = 0
     
        W,b = self.load_model()
        self.regressor = LogisticRegression(self.x, self.y, self.feature_size, self.label_size, W, b)
        self.predict = theano.function(inputs=[self.x], outputs=self.regressor.p_1)

    def load_model(self, fich="model.tkl"):
        with open(fich) as f:
            W, b = pickle.load(f)
            return (W,b)
            #self.regressor.reinit(self.x, self.y, W, b)
       
    def test(self):
        return self.predict(self.test_set)


class LogisticRegressionTrainer():
    def __init__(self, datasets, n_train_batches=1):
        """ datasets = (train_set, valid_set, test_set) """
        self.inputs, self.labels = datasets[0]  # inputs, labels from train_set
        self.datasets = datasets
        self.N, self.feature_size = self.inputs.shape
        self.batch_size = self.N / n_train_batches
        self.n_train_batches = n_train_batches
        self.label_size = self.labels.shape[1]
        self.training_steps = 1000
        self.learning_rate = 5.0

        # Declare Theano symbolic variables (x -> inputs, y-> labels)
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        W, b = self.load_model()
        self.regressor = LogisticRegression(self.x, self.y, self.feature_size, self.label_size, W, b)
        self.cross_error = theano.function(inputs=[self.x, self.y], outputs=self.regressor.cross_err)

        self.gw, self.gb = T.grad(self.regressor.cross_err, [self.regressor.W, self.regressor.b])             # Compute the gradient of the cost
        self.updates = (
                (self.regressor.W, self.regressor.W - self.learning_rate * self.gw),
                (self.regressor.b, self.regressor.b - self.learning_rate * self.gb))

        self.train = theano.function(
            inputs=[self.x, self.y],
            outputs=self.regressor.least_square,
            updates=self.updates)

        self.predict = theano.function(inputs=[self.x], outputs=self.regressor.p_1)

        self.test = theano.function(inputs=[self.x, self.y],outputs=self.regressor.least_square)
    
    def load_model(self, fich="model.tkl"):
        print "loading previous model"
        with open(fich) as f:
            W, b = pickle.load(f)
        return(W, b)

    def save_model(self):
        with open("model.tkl", 'w') as fich:
            pickle.dump(
                (self.regressor.W.get_value(), self.regressor.b.get_value()),
                fich
                )

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

        best_params = None
        best_validation_loss = np.inf
        test_score = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            #print [p.get_value() for p in self.regressor.params]
            for minibatch_index in xrange(self.n_train_batches):
                self.train(
                    self.inputs[minibatch_index * self.batch_size : (minibatch_index + 1) * self.batch_size],
                    self.labels[minibatch_index * self.batch_size : (minibatch_index + 1) * self.batch_size])
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
    trainer = LogisticRegressionTrainer(D, n_train_batches=50)
    trainer.train_loop()
    #print trainer.predict(D[0])
    print trainer.regressor.W.get_value(), trainer.regressor.b.get_value()



if __name__ == "__main__":
    main()

