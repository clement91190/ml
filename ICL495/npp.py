import numpy as np
from numpy.linalg import eig, inv  # , matrix_rank
from pca import get_U_for_PCA


def compute_graph_matrix(X, distance, heat_func, N=5, temp=5.0):
    S = np.zeros((X.shape[0], X.shape[0]))
    for i, x1_vect in enumerate(X):
        mins = []
        vals = []
        for j, x2_vect in enumerate(X):
            if len(mins) < N:
                mins.append(j)
                vals.append(distance(x1_vect, x2_vect))
                change = True
            else:
                d = distance(x1_vect, x2_vect)
                if d < vals[0]:
                    mins[0] = j
                    vals[0] = d
                    change = True
            if change:
                # simultaneous sorting
                vals, mins = [list(t) for t in zip(*sorted(zip(vals, mins)))]

        for j, d in zip(mins, vals):
            S[i, j] = heat_func(d, temp)
    return S


def get_D(S):
    D = np.diag(S.sum(1))
    return D


def get_U_for_NPP(X, compos=50):
    """ in this case, send the training and testing at the same time """

    N, F = X.shape
    print "...PCA to go to N dim space"
    M = N
    U = get_U_for_PCA(X, M)
    Xred = np.dot(X, U)

    euclid_sqr = lambda x1, x2: np.mean(np.square(x1 - x2))
    heat_func = lambda d, t: np.exp(- d / t)
    print "... compute graph matrix"
    S = compute_graph_matrix(Xred, distance=euclid_sqr, heat_func=heat_func)
    D = get_D(S)
    Xred = np.matrix(Xred)
    #print "rank of Xred : ", matrix_rank(Xred)
    #print "checking shapes ..."
    #print Xred.shape, D.shape
    #print " checking singular ? ", matrix_rank(Xred.transpose() * D * Xred)
    lamb, V = eig(inv(Xred.transpose() * (D) * Xred) * Xred.transpose() * (D - S) * Xred)
    #ind, _ = max(enumerate(lamb > 0.), key=lambda (i, bol): (bol, i))
    ind = lamb.argsort()
    W = V[:, ind[:compos]]
    return np.dot(U, W)
