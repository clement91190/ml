import cPickle
import numpy as np
#from logistic_reg import LogisticRegressionTester
from code.nn.NNTester import NNTester
from utils import complete 


def load_dataset(method, set):
    fich = 'data/' + method + '/' + set + 'ing_set.pkl'
    with open(fich) as f:
        print "loading file"
        if set == "test":
            (dataset, index) = cPickle.load(f)
        else:
            ((train, _, train_ind), (valid, _, valid_ind), (test, _, test_ind)) = cPickle.load(f)
            dataset = np.concatenate((train, valid, test))
            index = np.concatenate((train_ind, valid_ind, test_ind))
        return (dataset, index)


def get_results(dataset, index, method, learn_type):
    #dataset, index = load_dataset(method, set)
    tester = NNTester(dataset, method, learn_type)
    results = tester.test()
    return results, index


def write(dataset, index, method, learn_type, set):
    file = 'data/' + method + '/' + learn_type + '/results_' + set + '.csv'
    with open(file, 'w') as fich:
        first_ligne = """GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n"""
        fich.write(first_ligne)
        results, index = get_results(dataset, index, method, learn_type)
        print results[0]
        for i, res in enumerate(results):
            c = "{}".format(index[i])
            for proba in complete(res, learn_type):
                c += ",{}".format(proba)
            fich.write(c + "\n")


def main():
    methods = ["method_gray"]
    learn_types = ["global", "q1", "q2", "q3", "q4"]
    for m in methods:
        train_dataset, index = load_dataset(m, "train")
        for lt in learn_types:
            write(train_dataset, index, m, lt, "train")
        test_dataset, index = load_dataset(m, "test")
        for lt in learn_types:
            write(test_dataset, index, m, lt, "test")

if __name__ == "__main__":
    main()
