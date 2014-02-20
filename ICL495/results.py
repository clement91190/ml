from sklearn import neighbors
import numpy as np
from numpy.linalg import eig, inv, matrix_rank 
import matplotlib.pyplot as plt
from utils import center


""" OLD FILE NEW FILE ->>> Coursework 1 ipython """
  
def get_U_for_id(size=64):
    return np.eye(size * size)

def test_features(fea_train_red, fea_test_red, gnd_train, gnd_test):
    """ function to test the feature on a classification algorithm and return the error,
    """
    correct = []
    print "...training"
    size = gnd_test.shape[0]
    for i in range(0, fea_test_red.shape[1], 3):
        #print i
        X = fea_train_red[:,:i+1]
        Y = gnd_train
        n_neighbors = 2
        weights = 'distance'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, Y)
        prediction = clf.predict(fea_test_red[:, :i + 1])
        correct.append(np.sum(prediction - gnd_test == 0))
    correct = 1.0 * np.array(correct) / size
    print "correct", correct
    return 1.0 - correct


def prepare_data(train_idx, (train, fea, gnd)):
    """ in this function we can chose which method to apply i
        we return the training and testing low dimensional feature, 
        and associated class
    """
    test_idx = list(set(range(fea.shape[0])) - set(train_idx))
    fea_train = np.array([fea[i - 1] for i in train_idx])
    gnd_train = np.array([gnd[i - 1, 0] for i in train_idx]).flatten()
    gnd_train, fea_train = zip(*sorted(
        zip(gnd_train, fea_train),
        key=lambda (i, j): i))  # sorting per class 
    gnd_train, fea_train = np.array(gnd_train), np.array(fea_train)
    
    fea_test = np.array([fea[i - 1] for i in test_idx])
    gnd_test = np.array([gnd[i - 1] for i in test_idx]).flatten()
        
    #U = get_U_for_PCA(fea_train, 20)
    #U = get_U_for_NPP(fea_train, 300)
    #U = get_U_for_fast_ICA(fea_train, 40)
    U = get_U_for_whitened_PCA(fea_train, 70)
    #U = get_U_for_LDA(fea_train, gnd_train, 50)
    
    print "shape of U :", U.shape

    fea_train_red = np.dot(center(fea_train), U)
    fea_test_red = np.dot(center(fea_test), U)
    #print "size", fea_train_red.shape
    
    #print fea_train_red.shape
    fea_train_red, fea_test_red = center(fea_train_red), center(fea_test_red)
    return (fea_train_red, fea_test_red, gnd_train, gnd_test)


def main():
    train, fea, gnd = load_data()
    error = []
    #print train;
    
    for jj, train_idx in enumerate(train[:3]):
        data = prepare_data(train_idx, (train, fea, gnd))
        error.append(test_features(*data))
    print error
    plt.plot(np.mean(error, 0))
    plt.show()

if __name__ == "__main__":
    main()

error = []

