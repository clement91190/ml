import numpy as np
from pca import get_U_for_whitened_PCA


def decorellate(w, W, j):
    """ assuming that W is orthogonal, we delete the projection of w on this basis
    to prevent all the component to converge toward the same value """
    w -= np.dot(np.dot(w, W[:j].T), W[:j])
    return w


def get_U_for_fast_ICA(X, compos=20):
    N, F = X.shape
    print "...PCA to go to N dim space"
    M = N
    U = get_U_for_whitened_PCA(X, M)
    Xred = np.dot(X, U)
    M = Xred.shape[1]

    Xred = Xred.T  # to have the same notation than the paper
    g = np.tanh
    g_der = lambda y: np.ones(y.shape) - np.square(np.tanh(y))
    func = lambda y: (g(y), g_der(y))  # define the function and derivative

    W = np.zeros((M, M))
    w_init = np.random.random((M, M))  # init with random calues

    for j in range(M):
        w = w_init[j, :].copy()
        w /= np.sqrt((w ** 2).sum())
        w = np.matrix(w).T  # trick to have a column vector
        converged = False
        while not converged:
            gy, gdy = func(np.dot(w.T, Xred))
            wp = np.matrix((np.array(Xred) * np.array(gy)).mean(axis=1)).T - np.array(gdy.mean()) * np.array(w)
            decorellate(wp.T, W, j)  # gramshmidt like orhogonalization
            wp /= np.sqrt((np.array(wp) ** 2).sum())
            eps = np.abs(np.abs((wp.T * w).sum()) - 1)
            converged = eps < 0.0001
            w = wp
        W[j, :] = w.reshape(M)

    Mat = np.dot(U, W.T)
    #print Mat.shape
    return Mat[:, :compos]
