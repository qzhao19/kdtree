import numpy as np

class Node(object):
    def __init__(self, 
        
        # begin_index = -1, 
        # end_index = -1, 
        data = None,
        label = None,
        is_leaf = False, 
        partition_axis = -1, 
        # partition_value = np.Inf, 
        radius = -1, 
        left_child = None, 
        right_child = None):
        
        # self.begin_index = begin_index
        # self.end_index = end_index

        self.data = data
        self.label = label

        self.is_leaf = is_leaf
        self.partition_axis = partition_axis
        # self.partition_value = partition_value
        self.radius = radius

        self.left_child = left_child
        self.right_child = right_child



class KDTree(object):   
    def __init__(self, leaf_size = 40, metric = "euclidean"):
        
        # self.data = data
        # if self.data.size == 0:
        #     raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size
        self.metric = metric

        # self.n_samples = self.data.shape[0]
        # self.n_features = self.data.shape[1]

        # self.n_levels = int(np.log2(np.max(1, (self.n_samples - 1) / self.leaf_size)) + 1)
        # self.n_nodes = int(2 ** self.n_levels) - 1

        if metric == "euclidean":
            self.p = 2
        elif metric == "manhattan":
            self.p = 1
        # elif metric == "minkowski":
        #     self.p = 

        self.tmp = []


    def _build_tree(self, root, X, y):
        
        n_samples = X.shape[0]
        n_features = X.shape[1]

        lower_bounds = np.zeros(n_features)
        upper_bounds = np.zeros(n_features)
        range_bounds = np.zeros(n_features)

        for i in range(n_features):
            lower_bounds[i] = np.Inf
            upper_bounds[i] = -np.Inf
        
        # for i in range(n_samples):
        #     for j in range(n_features):
        #         lower_bounds[j] = min(lower_bounds[j], dataset[i, j])
        #         upper_bounds[j] = max(upper_bounds[j], dataset[i, j])

        lower_bounds = np.min(X, axis = 0)
        upper_bounds = np.max(X, axis = 0)

        
        # partition_axis = 0
        # for i in range(n_features):
        #     range_bounds[i] = upper_bounds[i] - lower_bounds[i]
        #     if range_bounds[i] > range_bounds[partition_axis]:
        #         partition_axis = i

        range_bounds = np.abs(upper_bounds - lower_bounds)
        partition_axis = np.argmax(range_bounds)











