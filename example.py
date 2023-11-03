import numpy as np
from sklearn import datasets
from kdtree import kd_tree


def test_kdtree(data):
    num_samples, _ = data.shape
    
    ratio = 0.85
    X_train = data[:int(ratio*num_samples), :]
    X_test = data[int(ratio*num_samples):, :]

    kdtree = kd_tree.KDTree(X_train)
    kdtree.build_tree()

    knn = kdtree.query(X_test, 4)
    for nn in knn:
        print(nn)

    rnn = kdtree.query_radius(X_test, 1.0)
    print("****************")
    for nn in rnn:
        print(nn)

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    test_kdtree(X)
