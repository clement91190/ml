import cPickle
from code.nn.multi_perceptron import MLP
from code.nn.conv_net import Lenet5

import theano
import theano.tensor as T


class NNTester():
    def __init__(self, test_set, method, learn_type):

        self.method = method
        self.learn_type = learn_type

        self.x = T.matrix("x")
        self.y = T.matrix("y")
        self.test_set = test_set
        self.batch_size, self.feature_size = self.test_set.shape

        params, architecture = self.load_model()
        self.neural_net = MLP(self.x, self.y, architecture, params)
        #self.neural_net = Lenet5(self.x, self.y, architecture, params)
        self.predict = theano.function(inputs=[self.x], outputs=self.neural_net.proba)

    def load_model(self):
        fich = 'data/' + self.method + '/' + self.learn_type + '/model.tkl'
        with open(fich) as f:
            params, architecture = cPickle.load(f)
            return (params, architecture)

    def test(self):
        return self.predict(self.test_set)


