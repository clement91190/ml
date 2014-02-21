import numpy as np
from numpy.linalg import pinv
import csv


methods = ["method_gray"]
learn_type = ["q1", "q2"]


def load_est(method, learn_type, set="train"):
    """ load the file with the results """
    est_k = []
    est_path = 'data/' + method + '/' + learn_type + '/results_' + set + '.csv'
    ids = []
    with open(est_path) as est_f:
        reader = csv.reader(est_f)
        for i, row in enumerate(reader):
            if i > 0:
                est_k.append(np.array(row[1:], dtype='float32'))
                ids.append(int(row[0]))
    est_k = np.array(est_k)  # shape(N, 37)
    return est_k, ids


def load_sol():
    sol_path = 'data/original_data/training_solutions_rev1.csv'
    sol = []
    with open(sol_path) as sol_f:
        reader = csv.reader(sol_f)
        for i, row in enumerate(reader):
            if i > 0:
                sol.append(np.array(row[1:], dtype='float32'))
    sol = np.array(sol)  # shape(N, 37)
    return sol


def save_results(results, index):
    file = 'data/results_fusion.csv'
    with open(file, 'w') as fich:
        first_ligne = """GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n"""
        fich.write(first_ligne)
        print results[0]
        for i, res in enumerate(results):
            c = "{}".format(index[i])
            for proba in res:
                c += ",{}".format(proba)
            fich.write(c + "\n")


def stat(est, sol):
    """ give estimations of the results. """
    res_per_colum = np.mean(np.abs(est - sol), 0)
    print " ... precision per answer..."
    print res_per_colum


def produce_fusion_res(T):
    est = []
    for m in methods:
        for lt in learn_type:
            est_k, ids = load_est(m, lt, "test")
            est.append(est_k)
    est = np.array(est)  # shape(K, N, 37)
    res = []
    print "...fusion for testing set"
    for d in range(37):
        est_d = est[:, :, d]  # shape : (K, N, 37) -> est_d(K,N)
        res_d = np.dot(T[d], est_d)  # shape(1, N)
    res.append(res_d[0, :])
    res = np.array(res)  # shape(37, N)
    res = res.T
    save_results(res, ids)


def results_on_subset(est, sol):
    """ shape(N, 37 ) """

    res = np.mean(np.sqrt(np.sum(np.square(est) - np.square(sol), 1)))
    print res


def main():
    est = []
    print "... loading results on train ..."
    sol = load_sol()
    for m in methods:
        for lt in learn_type:
#TODO add try
            est_k, ids = load_est(m, lt, "train")
            stat(est_k, sol)
            raw_input()
            est.append(est_k)
    est = np.array(est)
    T = []
    res_training = []
    print "... calculating pseudo inverse ..."
    for d in range(37):
        est_d = est[:, :, d]  # shape : (K, N, 37) -> est_d(K,N)
        sol_d = sol[:, d]  # shape(1, N)
        T_d = np.dot(sol_d, pinv(est_d))  # shape(1, K)
        res_d = np.dot(T_d, est_d)  # shape(1, N)
        res_training.append(res_d[0, :])
        T.append(T_d)
    res_training = np.array(res_training)
    results_on_subset(res_training, sol)
    print " ... saving results ..."
    produce_fusion_res(T)  # shape(37, 1,  K)

if __name__ == "__main__":
    main()
