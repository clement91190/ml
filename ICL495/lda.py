import numpy as np
from numpy.linalg import eig, inv 
from utils import get_cov_sample, make_sym
from pca import get_U_for_PCA


def get_U_for_LDA(X, gnd, compos):
    """ same as before, compute the projection matrix
    for LDA """
    N, F = X.shape
    NC = [len([i for i in gnd if i == j]) for j in range(min(gnd), max(gnd) + 1)]  # compute the number of elements per class
    print "...PCA to go to N-C dim space"
    M = N
    U = get_U_for_PCA(X, M)  # N-C ?
    Xred = np.dot(X, U)
    Cj = np.array([np.array([Xred[i] for i, g in enumerate(gnd) if g == j]) for j in range(min(gnd), max(gnd) + 1)])
    muj = Cj.mean(1)
    Sb = np.sum([ncj * np.matrix(muj[j, 0]).T * np.matrix(muj[j, 0]) for j, ncj in enumerate(NC)], 0)
    Sw = np.sum([get_cov_sample(c.T) for c in Cj], 0)
    Sw, Sb = np.matrix(Sw), np.matrix(Sb)
    lamb, Q = eig(make_sym(inv(Sw) * Sb))
    W = np.dot(U, np.real(Q[:, :compos]))
    #print "shape of W, :", W.shape
    return W
