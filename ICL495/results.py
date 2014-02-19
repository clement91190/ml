from scipy.io import loadmat
from sklearn import neighbors
import numpy as np
from numpy.linalg import eig, inv, matrix_rank 
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
    compos = min(compos, matrix_rank(cov))  # keep only useful info.
    #print cov.shape
    lamb, V = eig(cov)
    ind = (-lamb).argsort()
    ind = ind[:compos]
    print "loss : {} %".format(100.0 * (1 - np.sum(lamb[ind]) / np.sum(lamb)))
    return lamb[ind], np.matrix(V[:, ind])


def get_U_for_PCA(X, compos=20):
    print "..start analyses"
    lamb, V = PCA(X, compos) 
    temp = np.diag(1.0 / np.sqrt(lamb))  # compute inverse of square root (easy because diagonal matrix)
    U = X.transpose() * V * np.nan_to_num(temp)
    print "... done"
    #print (X.transpose() - np.dot(U, np.dot(X, U))).mean()

    #raw_input()
    #return U[:, :compos]
    return U


def get_U_for_whitened_PCA(X, compos=10):
    lamb, V = PCA(X, compos) 
    temp = np.diag(1.0 / lamb) 
    U = X.transpose() * V * temp 
    return U


def get_cov_sample(X):
    """ return the covariance of samples X """
    centered = np.matrix(center(X))
    return np.dot(centered, centered.transpose())


def get_U_for_LDA(X, gnd, compos):
    C = len(set(gnd))
    print "... {} classes".format(C)
    N, F = X.shape
    NC = [len([i for i in gnd if i==j]) for j in range(min(gnd), max(gnd) + 1)]
    print "...PCA to go to N-C dim space"
    M = N - 1
    U = get_U_for_PCA(X, M)  #N-C ? 
    Xred = np.dot(X, U)
    #print np.array([X[i] for i, g in enumerate(gnd) if g == 1 ])
    Cj = np.array([np.array([Xred[i] for i, g in enumerate(gnd) if g == j]) for j in range(min(gnd), max(gnd) + 1)])
    muj = Cj.mean(1)
    Sb = np.sum([ncj * np.matrix(muj[j,0]).transpose() * np.matrix(muj[j,0]) for j, ncj in enumerate(NC)], 0)
    #print Sb.shape
    Sw = np.sum([get_cov_sample(c.transpose()) for c in Cj], 0)
    #print Sw.shape
    Sw, Sb = np.matrix(Sw), np.matrix(Sb)
    lamb, Q = eig(make_sym(inv(Sw) * Sb))
    #print Q
    W = np.dot(U, np.real(Q[:,: compos]))
    print "shape of W, :", W.shape
    return W

def make_sym(A):
    return 0.5 * np.matrix(A + A.transpose())


def compute_graph_matrix(X, distance, heat_func, N=5, temp = 5.0):
    S = np.zeros((X.shape[0], X.shape[0]))
    print S.shape
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
    #print "rang, shape", matrix_rank(D), D.shape
    return D
    

def get_U_for_NPP(X, compos=50):
    """ in this case, send the training and testing at the same time """
   
    N, F = X.shape
    print "...PCA to go to N-C dim space"
    M = N 
    U = get_U_for_PCA(X, M)  # N-C ? 
    Xred = np.dot(X, U)

    euclid_sqr = lambda x1, x2: np.mean(np.square(x1 - x2))
    heat_func = lambda d, t : np.exp(-d/t)
    print "... compute graph matrix"
    S = compute_graph_matrix(Xred, distance=euclid_sqr, heat_func=heat_func)
    D = get_D(S)
    print S.shape, D.shape
    Xred = np.matrix(Xred)
    print "rank of Xred : ", matrix_rank(Xred)
    print "checking shapes ..."
    print Xred.shape, D.shape
    print " checking singular ? ", matrix_rank(Xred.transpose() * D * Xred)
    lamb, V = eig(inv(Xred.transpose() * (D) * Xred) * Xred.transpose() * (D - S) * Xred)
    #ind, _ = max(enumerate(lamb > 0.), key=lambda (i, bol): (bol, i))
    ind = lamb.argsort()
    W = V[ind[:compos]]
    print W.shape
    return np.dot(U, W.tranpose())


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
    for i in range(0, fea_test_red.shape[1], 3):
        #print i
        X = fea_train_red[:,:i+1]
        Y = gnd_train
        #print X.shape, Y.shape
        #print X
        #print Y
        n_neighbors = 2
        weights = 'distance'
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, Y)
        prediction = clf.predict(fea_test_red[:, :i + 1])
        #prediction = clf.predict(fea_train_red)
        #print "predshape",  prediction.shape
        #print "gnd_shape",  gnd_test.shape
        #print prediction - gnd_test == 0
        #print (prediction - gnd_test == 0).shape
        correct.append(np.sum(prediction - gnd_test == 0))
    correct = 1.0 * np.array(correct) / size
    print "correct", correct
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
        
    #U = get_U_for_PCA(fea_train, 20)
    #U = get_U_for_NPP(fea_train, 20)
    U = get_U_for_whitened_PCA(fea_train, 20)
    #U = get_U_for_LDA(fea_train, gnd_train, 50)
    #raw_input()
    print "shape of U :", U.shape
    print fea_train.shape
    print fea_test.shape

    fea_train_red = np.dot(center(fea_train), U)
    fea_test_red = np.dot(center(fea_test), U)
    #print "size", fea_train_red.shape
    
    print fea_train_red.shape
    #fea_train_red = fea_train
    #fea_test_red = fea_test
    #print fea_train_red
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

