import numpy as np

class KDTree2(object):
    def __init__(self, leaf_size = 10, splitter = None):
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

    def build_tree(self, data):
        """
        build a kd-tree for O(n log n) nearest neighbour search

        input:
            data:       2D ndarray, shape =(ndim,ndata), preferentially C order
            leafsize:   max. number of data points to leave in a leaf

        output:
            kd-tree:    list of tuples
        """
        num_samples, num_features = data.shape
        
        # find bounding hyper-rectangle
        hyper_rect = np.zeros((2, num_features))
        hyper_rect[0,:] = data.min(axis=0)
        hyper_rect[1,:] = data.max(axis=0)

        # create root of kd-tree
        partition_axis = self._find_partition_axis(data)
        indices = np.argsort(data[:, partition_axis], kind='mergesort')
        data[:,:] = data[indices, :]
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
                data[:,:] = data[indices, :]
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

        for t in self.tree:
            print(t)
        # return tree

    
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

        return ((c - centroid)**2).sum() < radius



