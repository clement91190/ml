import random
import cPickle
import numpy
import theano
import theano.tensor as T


class LogisticRegression():
    def __init__(self, input, n_in, n_out):
        """ input is a matrix containing a minibatch of vector """
        
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]  

    def rmse(self, proba):
        """ rmse over the minibatch """
        return T.mean(T.sqr(proba - self.p_y_given_x))


def learn():
    
    #parameters : 
    learning_rate = 0.13
    batch_size = 50    

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]


    #print test_set_y.dtype
    # compute number of minibatches for training, validation and testing
    print "number of images in train :", train_set_x.shape[0]
    n_train_batches = train_set_x.shape[0] / batch_size
    print "number of batches in train", n_train_batches
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size

    print "building the model"

    index = 0 # index to a [mini]batch
    y = T.fmatrix('y')
    x = T.fmatrix('x')
   

    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=37)
    cost = classifier.rmse(y)
    
    # compiling a Theano function that computes the mistakes that are made by
    validate_model = theano.function(
        inputs=[x, y],
        outputs=classifier.rmse(y),
       )
    
    test_model = theano.function(
        inputs=[x, y],
        outputs=classifier.rmse(y),
       )
     # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]


    train_model = theano.function(
        inputs=[x, y],
        updates=updates,
        outputs=classifier.rmse(y),
       )


  #givens={
  #          x: test_set_x[index * batch_size: (index + 1) * batch_size],
  #          y: test_set_y[index * batch_size: (index + 1) * batch_size]}  # the model on a minibatch
    
    x1 = test_set_x[index * batch_size: (index + 1) * batch_size]
    y1 = test_set_y[index * batch_size: (index + 1) * batch_size]  # the model on a minibatch
 
    while True:
        index = random.randint(0, n_train_batches - 1)
        x1 = test_set_x[index * batch_size: (index + 1) * batch_size]
        y1 = test_set_y[index * batch_size: (index + 1) * batch_size]  # the model on a minibatch
        print train_model(x1, y1)
   
    


def load_data():
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    print '... loading data'
    dataset = 'data/training_set_v1/training.pkl'
    # Load the dataset
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

        #test_set_x, test_set_y = shared_dataset(test_set)
    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y  # T.cast(shared_y, 'int32')


