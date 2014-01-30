import cPickle
#from logistic_reg import LogisticRegressionTester
from interface import NNTester


def load_dataset():
    with open("data/testing_set_v1.tkl") as f:
        print "loading file"
        (dataset, index) = cPickle.load(f)
        return (dataset, index)

def get_results():
    dataset, index = load_dataset() 
    #tester = LogisticRegressionTester(dataset)
    tester = NNTester(dataset)

    results = tester.test()
    return results, index

def write(file="results.csv"):
    with open(file, 'w') as fich:
        first_ligne = """GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n"""
        fich.write(first_ligne)
        results, index = get_results()
        print results[0]
        for i, res in enumerate(results):
            c = "{}".format(index[i])
            for proba in res:
                c += ",{}".format(proba)
            fich.write(c+"\n")

def main():
    write()

if __name__ == "__main__":
    main()
