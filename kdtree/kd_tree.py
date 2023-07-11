import copy
import numpy as np

class KDTree(object):
    def __init__(self, data, leaf_size = 10):
        self.data = data

        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size

        self.root = []

    def _find_partition_axis(self, data):        
        lower_bounds = np.min(data, axis = 0)
        upper_bounds = np.max(data, axis = 0)

        range_bounds = np.abs(upper_bounds - lower_bounds)
        partition_axis = np.argmax(range_bounds)
        return partition_axis

    
    def build_tree(self):

        data = copy.deepcopy(self.data)
        n_samples = data.shape[0]
        n_features = data.shape[1]

        # find bounding hyper-rectangle
        hyper_rect = np.zeros((2, n_features))
        hyper_rect[0, :] = data.min(axis = 0)
        hyper_rect[1, :] = data.max(axis = 0)

        # create root 
        partition_axis = self._find_partition_axis(data)
        indices = np.argsort(data[:, partition_axis], kind='mergesort')

        data[:, :] = data[indices, :]
        mid_val = data[n_samples // 2, partition_axis]

        left_hyper_rect = hyper_rect.copy()
        right_hyper_rect = hyper_rect.copy()

        left_hyper_rect[1, 0] = mid_val
        right_hyper_rect[0, 0] = mid_val

        # data, indices, left_hyper_rect, right_hyper_rect, left_child_index, right_child_index
        self.root = [(None, None, left_hyper_rect, right_hyper_rect, None, None)]

        # data, indices, depth, parent index, is_left_child
        stack = [(data[:n_samples//2, :], indices[:n_samples//2], 1, 0, True),
                (data[n_samples//2:, :], indices[n_samples//2:], 1, 0, False)]

        while stack:

            # pop data off stack
            data, indices, depth, index, is_left_child = stack.pop()
            n_samples = data.shape[0]
            node_ptr = len(self.root)

            _data, _indices, _left_hyper_rect, _right_hyper_rect, left_child, right_child = self.root.pop()

            if is_left_child:
                self.tree[index] = _data, _indices, _left_hyper_rect, _right_hyper_rect, node_ptr, right_child
            else:
                self.tree[index] = _data, _indices, _left_hyper_rect, _right_hyper_rect, left_child, node_ptr
            
            # leaf node?
            if n_samples < self.leaf_size:
                _data = data
                _indices = indices
                leaf = (_data, _indices, None, None, 0, 0)
                self.root.append(leaf)
            else:
                partition_axis = self._find_partition_axis(data)
                cur_indices = np.argsort(data[:,partition_axis], kind='mergesort')
                data[:,:] = data[cur_indices, :]
                indices = indices[cur_indices]
                node_ptr = len(self.root)







