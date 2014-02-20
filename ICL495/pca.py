import numpy as np
from utils import center
from numpy.linalg import eig, matrix_rank


def PCA(X, compos):
    """ return the compos highest eigenvalues and corresponding eigenvectors,
    of the covariance matrix of X (using the XX.T trick) """
    X = center(X)
    X = np.matrix(X)
    cov = np.dot(X, X.T)
    compos = min(compos, matrix_rank(cov))  # keep only useful info.
    lamb, V = eig(cov)
    ind = (-lamb).argsort()  # sort the eigenvalues
    ind = ind[:compos]  # keep only the compos best
    #print "loss : {} %".format(100.0 * (1 - np.sum(lamb[ind]) / np.sum(lamb)))  # print the loss of variance.
    return lamb[ind], np.matrix(V[:, ind])


def get_U_for_PCA(X, compos=20):
    """ compute the U matrix (linear transformation matrix)
    to go from feature space to low dimensional space """
    print "..start analyses"
    lamb, V = PCA(X, compos)
    temp = np.diag(1.0 / np.sqrt(lamb))  # compute inverse of square root (easy because diagonal matrix)
    U = X.T * V * temp
    print "... done"
    return U


def get_U_for_whitened_PCA(X, compos=10):
    """ same but for whitened PCA. """
    lamb, V = PCA(X, compos)
    temp = np.diag(1.0 / lamb)
    U = X.T * V * temp
    return U
