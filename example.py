import numpy as np
from kdtree import KDTree
from sklearn import datasets

def test(data):
    num_samples, _ = data.shape
    
    ratio = 0.85
    X_train = data[:int(ratio*num_samples), :]
    X_test = data[int(ratio*num_samples):, :]

    kd_tree = KDTree(X_train)
    kd_tree.build_tree()

    nn = kd_tree.query(X_test, 4)

if __name__ == "__main__":
    test(datasets)
