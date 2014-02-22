import cPickle
import os
from NNTrainer import NNTrainer
from utils import filters


def select((tx, ty, ids), learn_type):
    """ select a subset of the labels to learn only some of them ..."""
    if learn_type == "global":
        return (tx, ty)
    return (tx, ty[:, filters[learn_type]])


def load_data(learn_type, method):
    print '... loading data'
    dataset = 'data/' + method + '/training_set.pkl'
    # Load the dataset
    with open(dataset, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
    train_set = select(train_set, learn_type)
    valid_set = select(valid_set, learn_type)
    test_set = select(test_set, learn_type)
    print "... done "   
    return (train_set, valid_set, test_set)


def main():
    method = "method_gray_center"
    learn_type = "global"
    try:
        os.mkdir('data/' + method + '/' + learn_type)
    except:
        pass
    learning_rate = 0.01
    train, validate, test = load_data(learn_type, method)
   # N, feats, n_out = 100, 10, 2
   # train = (rng.randn(N, feats), 0.5 * np.ones((N, n_out)))
   # validate = (rng.randn(N, feats), 0.5 * np.ones((N, n_out)))
   # test = (rng.randn(N, feats), 0.5 * np.ones((N, n_out)))
    D = (train, validate, test)
    trainer = NNTrainer(D, method, learn_type, load=True, learning_rate=learning_rate, n_train_batches=50)
    trainer.train_loop()
    #print trainer.predict(D[0])
    print trainer.neural_net.params

if __name__ == "__main__":
    main()

