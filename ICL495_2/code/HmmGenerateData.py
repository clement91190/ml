import numpy as np


def hmm_gen_multinomial(nums, T, pi, A, E):
    S = np.zeros((nums, T), dtype=int)
    Y = np.zeros((nums, T), dtype=int)

    for i in range(nums):
        S[i, 0] = np.sum(np.random.random() > np.cumsum(pi))
        Y[i, 0] = np.sum(np.random.random() > np.cumsum(E[S[i, 0], :]))
              # Generate the rest of the observations.
        for l in range(1, T):
            S[i, l] = np.sum(np.random.random() > np.cumsum(A[S[i, l - 1], :]))
            Y[i, l] = np.sum(np.random.random() > np.cumsum(E[S[i, l], :]))
    return (Y, S)


def hmm_gen_ar1(nums, T, pi, A, E):
    S = np.zeros((nums, T), dtype=int)
    Y = np.zeros((nums, T), dtype=int)

    for i in range(nums):
        S[i, 0] = np.sum(np.random.random() > np.cumsum(pi))
        Y[i, 0] = np.random.normal() * 0.01

        # Generate the rest of the observations.
        for l in range(1, T):
            S[i, l] = np.sum(np.random.random() > np.cumsum(A[S[i, l - 1], :]))
            Y[i, l] = E[S[i, l]] * Y[i, l - 1] + np.random.normal() * 0.01
    return (Y, S)


def hmm_gen_normal(nums, T, pi, A, E):
    S = np.zeros((nums, T), dtype=int)
    Y = np.zeros((nums, T), dtype=int)

    for i in range(nums):
        S[i, 0] = np.sum(np.random.random() > np.cumsum(pi))
        Y[i, 0] = np.random.normal(E['mu'][S[i, 0]], np.sqrt(E['sigma2'][S[i, 0]]))
            # Generate the rest of the observations.
        for l in range(1, T):
            S[i, l] = np.sum(np.random.random() > np.cumsum(A[S[i, l - 1], :]))
            Y[i, l] = np.random.normal(E['mu'][S[i, l]], np.sqrt(E['sigma2'][S[i, l]]))
    return (Y, S)


def hmm_generate_data(nums, T, pi, A, E, outModel='multinomial'):
    return {
        'multinomial': hmm_gen_multinomial,
        'ar1': hmm_gen_ar1,
        'normal': hmm_gen_normal}[outModel](nums, T, pi, A, E)


def normalize_rows(M):
    """ useful function when dealing with probabilities """
    return (M.T / np.sum(M, 1)).T


class HMM:
    def __init__(self, Y, n_state, n_letters):
        self.Y = Y
        self.n_state = n_state
        self.n_letters = n_letters
        self.N, self.T = Y.shape  # number of sequence and length of the sequence
        self.random_init()

    def random_init(self):
        """random initialization of parameters
            E -> emission probability
            A -> state changes
            S -> states
        """
        E = np.random.random((self.n_state, self.n_letters))
        self.E = normalize_rows(E)
        #self.logE = np.log(E)  # we work with log values directly
        self.A = np.random.random((self.n_state, self.n_state))
        #self.logA = np.log(normalize_rows(A))  # trans matrix
        #self. = np.log(np.random.random((self.N, self.T)))  # init state
        self.pi = np.random.random(self.n_state)

    def cheat_init(self, E, A, pi):
        self.E = E
        self.A = A
        self.pi = pi

    def alpha(self, obs):
        """ filtering """
        alpha = np.zeros((self.T, self.n_state))  # keep the result in an array for faster recursion
        alpha[0, :] = self.pi * self.E[:, int(obs[0])]
        for k, o in enumerate(obs[1:]):
            o = int(o)
            alpha[k + 1, :] = self.E[:, o] * np.dot(self.A.T, alpha[k, :])
        return alpha

    def beta(self, obs):
        """ backward pass """
        beta = np.ones((self.T, self.n_state))  # keep the result in an array for faster recursion
        for k, o in enumerate(reversed(obs[1:])):
            o = int(o)
            beta[self.T - (k + 2), :] = np.dot(self.A, beta[self.T - (k + 1), :] * self.E[:, o])
        return beta

    def ksi(self, alpha, beta, obs, norm):
        """ probability : p(zi, zi-1 | obs)
            norm is the sum of gamma to normalize the data"""
        ksi = []
        for i, o in enumerate(obs[1:]):
            ksi.append(np.array([alpha[i, :] for j in range(self.n_state)]) * self.A * np.array([beta[i + 1, :] * self.E[:, o] for j in range(self.n_state)]).T / norm)
            #print 1, np.array([alpha[i, :] for j in range(self.n_state)])
            #print 2, self.A
            #print o
            #print "b2", self.E[:, o]
            #print 3,  np.array([beta[i + 1, :] * self.E[:, o] for j in range(self.n_state)]).T
            #print "res", ksi[-1]
            #raw_input()

        return np.array(ksi)

    def expect_max(self, debug=False):

        Es = []
        pis = []
        As = []
        for obs in self.Y:
            a = self.alpha(obs)
            norm = np.sum(a[-1, :])
            b = self.beta(obs)
            gamma = a * b
            gamma = (gamma.T / norm).T
            ksi = self.ksi(a, b, obs, norm)

            if debug:
                print "a",  a
                raw_input()

                print "norm", norm
                raw_input()

                print "b",  b
                raw_input()

                print "gamma", gamma
                raw_input()

                print "ksi", ksi
                raw_input()

            pis.append(gamma[0])
            As.append(normalize_rows(np.sum(ksi, 0)))
            for k in range(self.n_letters):
                self.E[:, k] = np.sum((self.Y[0, :] == k) * gamma.T, 1)
            Es.append(normalize_rows(self.E))

        print "pi", np.mean(pis, 0)
        print "A", np.mean(As, 0)
        print "E", np.mean(Es, 0)

        self.pi = np.mean(pis, 0)
        self.A = np.mean(As, 0)
        self.E = np.mean(Es, 0)
        #raw_input()

    def viterbi_algo(self, obs):
        """ obs is the sequence of observation ( inspired from the version on the Wikipedia page. )"""
        proba = []
        proba_old = []
        path = []
        path_old = []

        proba_old = [self.pi[s] * self.E[s, obs[0]] for s in range(self.n_state)]  # init with the probability of having s responsible for obs 0
        path_old = [[s] for s in range(self.n_state)]  # init with start state

        for o in obs[1:]:
            proba = []
            path = []
            for s in range(self.n_state):
                (prob, state) = max((proba_old[s0] * self.A[s0, s] * self.E[s, o], s0) for s0 in range(self.n_state))  # find the previous state with the heighest probability
                proba.append(prob)  # keep track of the probability
                path.append(path_old[state] + [s])  # add the state to the path
            path_old = path
            proba_old = proba
        (prob, state) = max((proba[s], s) for s in range(self.n_state))
        return (prob, path[state])
