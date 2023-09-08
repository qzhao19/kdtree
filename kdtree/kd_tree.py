import copy
import numpy as np

class KDTree(object):
    def __init__(self, data, leaf_size = 10, splitter = None):

        self.data = data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        num_samples = self.data.shape[0]
        num_features = self.data.shape[1]

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size
        self.tree = []

        self.splitter = splitter


    def _find_partition_axis(self, data):
        lower_bounds = np.min(data, axis = 0)
        upper_bounds = np.max(data, axis = 0)
        range_bounds = np.abs(upper_bounds - lower_bounds)
        partition_axis = np.argmax(range_bounds)
        return partition_axis

    def build_tree(self):
        """
        build a kd-tree for O(n log n) nearest neighbour search

        input:
            data:       2D ndarray, shape =(ndim,ndata), preferentially C order
            leafsize:   max. number of data points to leave in a leaf

        output:
            kd-tree:    list of tuples
        """
        data = copy.deepcopy(self.data)
        num_samples, num_features = data.shape
        # find bounding hyper-rectangle
        hyper_rect = np.zeros((2, num_features))
        hyper_rect[0,:] = data.min(axis=0)
        hyper_rect[1,:] = data.max(axis=0)

        # create root of kd-tree
        partition_axis = self._find_partition_axis(data)
        indices = np.argsort(data[:, partition_axis], kind='mergesort')
        data = data[indices, :]
        partition_val = data[num_samples//2, partition_axis]

        left_hyper_rect = hyper_rect.copy()
        right_hyper_rect = hyper_rect.copy()
        left_hyper_rect[1, 0] = partition_val
        right_hyper_rect[0, 0] = partition_val
        
        self.tree = [(None, None, left_hyper_rect, right_hyper_rect, None, None)]
        
        stack = [(data[:num_samples//2,:], indices[:num_samples//2], 1, 0, True),
                (data[num_samples//2:,:], indices[num_samples//2:], 1, 0, False)]

        # recursively split data in halves using hyper-rectangles:
        while stack:
            # pop data off stack
            data, data_indices, depth, parent, is_left = stack.pop()
            num_samples = data.shape[0]
            node_ptr = len(self.tree)

            # update parent node
            _data_indices, _data, _left_hyper_rect, _right_hyper_rect, left, right = self.tree[parent]
            if is_left:
                self.tree[parent] = (_data_indices, _data, _left_hyper_rect, _right_hyper_rect, node_ptr, right) 
            else:
                self.tree[parent] = (_data_indices, _data, _left_hyper_rect, _right_hyper_rect, left, node_ptr)

            # insert node in kd-tree
            # leaf node?
            if num_samples <= self.leaf_size:
                _data_indices = data_indices.copy()
                _data = data.copy()
                leaf = (_data_indices, _data, None, None, 0, 0)
                self.tree.append(leaf)

            # not a leaf, split the data in two      
            else:                  
                # splitdim = depth % ndim
                partition_axis = self._find_partition_axis(data)
                indices = np.argsort(data[:, partition_axis], kind='mergesort')
                data = data[indices, :]
                data_indices = data_indices[indices]
                node_ptr = len(self.tree)

                stack.append((data[:num_samples//2, :], data_indices[:num_samples//2], depth+1, node_ptr, True))
                stack.append((data[num_samples//2:, :], data_indices[num_samples//2:], depth+1, node_ptr, False))

                partition_val = data[num_samples//2, partition_axis]
                
                if is_left:
                    left_hyper_rect = _left_hyper_rect.copy()
                    right_hyper_rect = _left_hyper_rect.copy()
                else:
                    left_hyper_rect = _right_hyper_rect.copy()
                    right_hyper_rect = _right_hyper_rect.copy()
                
                left_hyper_rect[1, partition_axis] = partition_val
                right_hyper_rect[0, partition_axis] = partition_val
                # append node to tree
                self.tree.append((None, None, left_hyper_rect, right_hyper_rect, None, None))

    def _check_intersection(self, hyper_rect, centroid, radius):
        """
        checks if the hyperrectangle hrect intersects with the
        hypersphere defined by centroid and r2
        """
        upper_bounds = hyper_rect[1, :]
        lower_bounds = hyper_rect[0, :]
        c = centroid.copy()

        idx = c < lower_bounds
        c[idx] = lower_bounds[idx]

        idx = c > upper_bounds
        c[idx] = upper_bounds[idx]

        return ((c-centroid)**2).sum() < radius

    def _compute_dist(self, data, leaf_indices, leaf_data, k):
        """ find K nearest neighbours of data among ldata """
        num_samples, num_features = leaf_data.shape
        if k >= num_samples:
            k = num_samples

        dist = ((leaf_data - data[:num_samples,:])**2).sum(axis=1)
        indices = np.argsort(dist, kind='mergesort')
        return list(zip(dist[indices[:k]], leaf_indices[indices[:k]]))


    def _query_single_data(self, data, K):
        """ find the k nearest neighbours of datapoint in a kdtree """
        # root = self.tree
        stack = [self.tree[0]]
        knn = [(np.inf, None)]*K
        single_data = data[0, :]
        while stack:
            leaf_idx, leaf_data, left_hyper_rect, \
                    right_hyper_rect, left, right = stack.pop()

            # leaf
            if leaf_idx is not None:
                _knn = self._compute_dist(data, leaf_idx, leaf_data, K)
                if _knn[0][0] < knn[-1][0]:
                    knn = sorted(knn + _knn)[:K]

            # not a leaf
            else:
                # check left branch
                if self._check_intersection(left_hyper_rect, single_data, knn[-1][0]):
                    stack.append(self.tree[left])

                # chech right branch
                if self._check_intersection(right_hyper_rect, single_data, knn[-1][0]):
                    stack.append(self.tree[right])              
        return knn
    

    def query(self, data, K):
        """ find the K nearest neighbours for data points in data,
            using an O(n log n) kd-tree """

        num_samples, _ = data.shape
        # search kdtree
        knn = []
        for i in np.arange(num_samples):
            # _data = data[:,i].reshape((param,1)).repeat(leafsize, axis=1)
            _data = data[i, :][np.newaxis].repeat(self.leaf_size, axis=0)
            # print(_data)
            _knn = self._query_single_data(_data, K)
            knn.append(_knn)

        return knn

    def _query_radius_single_data(self, data, radius):
        # root = self.tree
        stack = [self.tree[0]]
        inside = []
        while stack:
            leaf_idx, leaf_data, left_hyper_rect, \
                    right_hyper_rect, left, right = stack.pop()

            # leaf
            if leaf_idx is not None:
                num_features = leaf_data.shape[1]
                distance = np.sqrt(((leaf_data - data.reshape((1,num_features)))**2).sum(axis=0))
                nn = np.where(distance<=radius)
                if len(nn[0]):
                    idx = leaf_idx[nn]
                    distance = distance[nn]
                    inside += (zip(distance, idx))

            else:
                if self._check_intersection(left_hyper_rect, data, radius):
                    stack.append(self.tree[left])

                if self._check_intersection(right_hyper_rect, data, radius):
                    stack.append(self.tree[right])              
        return inside
    
    def query_radius(self, data, radius):
        num_samples, _ = data.shape
        # search kdtree
        knn = []
        for i in np.arange(num_samples):
            # _data = data[:,i].reshape((param,1)).repeat(leafsize, axis=1)
            _data = data[i, :][np.newaxis].repeat(self.leaf_size, axis=0)
            # print(_data)
            _knn = self._query_radius_single_data(_data, radius)
            knn.append(_knn)

        return knn
