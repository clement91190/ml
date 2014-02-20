import numpy as np
from scipy.io import loadmat


def make_sym(A):
    return 0.5 * np.matrix(A + A.T)


def load_data(db="PIE"):
    """ load the data from the matlab files. """
    print " ...loading data"
    if db == "PIE":
        db += "/"
        train = loadmat('data/' + db + 'Train5_64.mat')['Train5_64']
        fea = loadmat('data/' + db + 'fea64.mat')['fea64']
        gnd = loadmat('data/' + db + 'gnd64.mat')['gnd64']
    else:
        db += "/"
        data = loadmat('data/' + db + 'YaleB_32x32.mat')
        fea = data['fea']
        gnd = data['gnd']
        train = []
        for i in range(50):
            t = loadmat('data/' + db + '5Train/{}.mat'.format(i + 1))
            train.append([e[0] for e in t['trainIdx']])
        train = np.array(train)
        train.reshape((train.shape[0], train.shape[1]))
    return (train, fea, gnd)


def center(fea_array):
    "center the data, data should be of shape (Nsamples, Features)"
    return fea_array - np.mean(fea_array, 0)


def get_cov_sample(X):
    """ return the covariance of samples X as a Matrix"""
    centered = np.matrix(center(X))
    return centered * centered.T


