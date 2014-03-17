from HmmGenerateData import hmm_generate_data, HMM
import numpy as np


def main():
    N = 100         # number of sequences
    T = 4        # length of the sequence
    pi = np.array([0.5, 0.5])  # inital probability pi_1 = 0.5 and pi_2 =0.5
    n_state = 2
    n_letters = 2

#two states hence A is a 2X2 matrix
    A = np.array([[0.5, 0.5], [0.5, 0.5]])  # p(y_t|y_{t-1})

#alphabet of 6 letters (e.g., a die with 6 sides) E(i,j) is the
#E = [1/6 1/6 1/6 1/6 1/6 1/6;      %p(x_t|y_{t})
#    1/10 1/10 1/10 1/10 1/10 1/2];
#    E = np.array([[995.0 / 1000] + [1.0 / 1000 for i in range(5)], [1.0 / 100 for i in range(5)] + [95.0 / 100.0]])
    E = np.array([[0.5, 0.5], [0.75, 0.25]])

    print "pi: {}".format(pi)
    print "A: {}".format(A)
    print "E: {}".format(E)

    [Y, S] = hmm_generate_data(N, T, pi, A, E)

#Y is the set of generated observations
#S is the set of ground truth sequence of latent vectors
    #print Y.shape
    #print S.shape
    #print Y[0]
    #print S[0]
    #Y = np.array([[0, 0, 1, 1]])
    print Y

#print Y[0]
#print S[0]

    hmm = HMM(Y, n_state, n_letters)
    hmm.cheat_init(E, A, pi)
    for i in range(100):
        hmm.expect_max()
        #print hmm.A
        #print hmm.E
        #print hmm.pi
        raw_input()

if __name__ == "__main__":
    main()
