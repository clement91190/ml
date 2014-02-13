from scipy.io import loadmat
from sklearn import neighbors
import numpy as np
from numpy.linalg import eig 
import matplotlib.pyplot as plt


def load_data():
    """ load the data from the matlab files. """
    print " ...loading data"
    db = "PIE/"
    #db = "PIE/"
    train = loadmat('data/' + db + 'Train5_64.mat')['Train5_64']
    fea = loadmat('data/' + db + 'fea64.mat')['fea64']
    gnd = loadmat('data/' + db + 'gnd64.mat')['gnd64']
    return (train, fea, gnd)


def PCA(X, compos):
    X = center(X)
    X = np.matrix(X)
    cov = np.dot(X, X.transpose())
    #print cov.shape
    lamb, V = eig(cov)  # perform eigenanalyses (sorted by eigen values)
    print "loss : {} %".format(100.0 * (1 - np.sum(lamb[:compos]) / np.sum(lamb)))
    return lamb, V


def get_U_for_PCA(X, compos=10):
    print "..start analyses"
    lamb, V = PCA(X, compos) 
    temp = np.diag(1.0 / np.sqrt(lamb))  # compute inverse of square root (easy because diagonal matrix)
    U = X.transpose() * V * temp
    print "... done"
    print (X.transpose() - np.dot(U, np.dot(X, U))).mean()

    raw_input()
    return U[:, :compos]


def get_U_for_whitened_PCA(X, compos=10):
    lamb, V = PCA(X, compos) 
    temp = np.diag(1.0 / lamb) 
    U = X.transpose() * V * temp 
    return U[:, :compos]


def get_U_for_LDA(X):
    pass


def get_U_for_NPP(X):
    pass


def get_U_for_fast_ICA(X):
    pass


def get_U_for_id(size=64):
    return np.eye(size * size)


def center(fea_array):
    return fea_array - np.mean(fea_array, 0)


def test_features(fea_train_red, fea_test_red, gnd_train, gnd_test):
    correct = []
    print "...training"
    size = gnd_test.shape[0]
    for i in range(0, fea_test_red.shape[1], 1):
        #print i
        X = fea_train_red[:i + 1]
        Y = gnd_train[:i + 1]
        n_neighbors = 1
        weights = 'distance'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, Y)
        prediction = clf.predict(fea_test_red)
        #print "predshape",  prediction.shape
        #print "gnd_shape",  gnd_test.shape
        # raw_input()
        #print prediction - gnd_test == 0
        #print (prediction - gnd_test == 0).shape
        correct.append(np.sum(prediction - gnd_test == 0))
    correct = 1.0 * np.array(correct) / size
    print correct
    return 1.0 - correct


def prepare_data(train_idx, (train, fea, gnd)):
    test_idx = list(set(range(fea.shape[0])) - set(train_idx))
    fea_train = np.array([fea[i - 1] for i in train_idx])
    gnd_train = np.array([gnd[i - 1, 0] for i in train_idx]).flatten()
    gnd_train, fea_train = zip(*sorted(
        zip(gnd_train, fea_train),
        key=lambda (i, j): i))  # sorting per class 
    gnd_train, fea_train = np.array(gnd_train), np.array(fea_train)
    
    fea_test = np.array([fea[i - 1] for i in test_idx])
    gnd_test = np.array([gnd[i - 1] for i in test_idx]).flatten()
        
    U = get_U_for_PCA(fea_train)
    #U = get_U_for_whitened_PCA(fea_train)
    #print U.shape
    #print fea_train.shape
    #print fea_test.shape

    fea_train_red = np.dot(center(fea_train), U)
    fea_test_red = np.dot(center(fea_test), U)
    
    #fea_train_red = fea_train
    #fea_test_red = fea_test
    #print fea_train_red
    fea_train_red, fea_test_red = center(fea_train_red), center(fea_test_red)
    return (fea_train_red, fea_test_red, gnd_train, gnd_test)


def main():
    train, fea, gnd = load_data()
    error = []
    #print train;
    
    for jj, train_idx in enumerate(train[:5]):
        data = prepare_data(train_idx, (train, fea, gnd))
        error.append(test_features(*data))
    print error
    plt.plot(np.mean(error, 0))
    plt.show()

if __name__ == "__main__":
    main()

error = []

