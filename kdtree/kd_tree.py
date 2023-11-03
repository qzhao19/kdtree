#!/bin/env python

import copy
import heapq
import numpy as np
from .kd_tree_node import KDTreeNode

class KDTree(object):
    """
    Parameters:
    ----------
    data: 2D ndarray
        shape =(ndata, ndim), preferentially C order
    
    leaf_size: int, default is 10
        max number of data points to leave in a leaf
    """
    def __init__(self, data, leaf_size = 10, metric = None):

        self.data = data
        if self.data.size == 0:
            raise ValueError("X is an empty array")

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")
        self.leaf_size = leaf_size
        self.tree = []

        if not metric:
            self.metric = "euclidean"
        else:
            self.metric = metric

    def _find_partition_axis(self, data):
        lower_bounds = np.min(data, axis = 0)
        upper_bounds = np.max(data, axis = 0)
        range_bounds = np.abs(upper_bounds - lower_bounds)
        partition_axis = np.argmax(range_bounds)
        return partition_axis

    def build_tree(self):
        """build a kd-tree for O(n log n) nearest neighbour search
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
        
        node = KDTreeNode(data=None, 
            indices=None, 
            left_hyper_rect=left_hyper_rect, 
            right_hyper_rect=right_hyper_rect, 
            left = None, 
            right=None
        )
        self.tree.append(node)
        
        stack = [(data[:num_samples//2,:], indices[:num_samples//2], 1, 0, True),
                (data[num_samples//2:,:], indices[num_samples//2:], 1, 0, False)]

        # recursively split data in halves using hyper-rectangles:
        while stack:
            # pop data off stack
            data, data_indices, depth, parent, is_left = stack.pop()
            num_samples = data.shape[0]
            node_ptr = len(self.tree)

            # update parent node
            cur_node = self.tree[parent]
            if is_left:
                cur_node.left = node_ptr
            else:
                cur_node.right = node_ptr

            # insert node in kd-tree
            # leaf node?
            if num_samples <= self.leaf_size:
                leaf_node = KDTreeNode(data=data.copy(), 
                    indices=data_indices.copy(), 
                    left_hyper_rect=None, 
                    right_hyper_rect=None, 
                    left=0, 
                    right=0,
                )
                self.tree.append(leaf_node)

            # not a leaf, split the data in two      
            else:                  
                # split on median to create a tree 
                partition_axis = self._find_partition_axis(data)
                indices = np.argsort(data[:, partition_axis], kind='mergesort')
                data = data[indices, :]
                data_indices = data_indices[indices]
                node_ptr = len(self.tree)

                stack.append((data[:num_samples//2, :], data_indices[:num_samples//2], depth+1, node_ptr, True))
                stack.append((data[num_samples//2:, :], data_indices[num_samples//2:], depth+1, node_ptr, False))

                partition_val = data[num_samples//2, partition_axis]
                
                if is_left:
                    left_hyper_rect = cur_node.left_hyper_rect.copy()
                    right_hyper_rect = cur_node.left_hyper_rect.copy()
                else:
                    left_hyper_rect = cur_node.right_hyper_rect.copy()
                    right_hyper_rect = cur_node.right_hyper_rect.copy()
                
                left_hyper_rect[1, partition_axis] = partition_val
                right_hyper_rect[0, partition_axis] = partition_val
                # append node to tree
                branch_node = KDTreeNode(data=None,
                    indices=None, 
                    left_hyper_rect=left_hyper_rect, 
                    right_hyper_rect=right_hyper_rect, 
                    left=0, 
                    right=0,
                )
                self.tree.append(branch_node)

    def _check_intersection(self, hyper_rect, centroid, radius):
        """
        checks if the hyperrectangle hrect intersects with the
        hypersphere defined by centroid and r2
        """
        upper_bounds = hyper_rect[1, :]
        lower_bounds = hyper_rect[0, :]
        c = centroid.copy()

        for i in range(len(c)):
            if c[i] > upper_bounds[i]:
                c[i] = upper_bounds[i]
            if c[i] < lower_bounds[i]:
                c[i] = lower_bounds[i]

        return np.sqrt(((c-centroid)**2).sum()) < radius

    def _compute_dist(self, data, leaf_indices, leaf_data, k):
        """ find K nearest neighbours of data among ldata """
        nn = []
        num_samples, num_features = leaf_data.shape
        if k >= num_samples:
            k = num_samples

        data = data[np.newaxis]
        dist = np.sqrt(((leaf_data - data)**2).sum(axis=1))
        indices = np.argsort(dist, kind='mergesort')
        for i in range(k):
            heapq.heappush(nn, (dist[indices[i]], leaf_indices[indices[i]]))
        return nn

    def _query_single_data(self, data, k):
        """ find the k nearest neighbours of datapoint in a kdtree """
        stack = [self.tree[0]]
        knn = []
        heapq.heappush(knn, (np.inf, 0))
        while stack:
            node = stack.pop()
            # leaf
            if node.indices is not None:
                single_knn = self._compute_dist(data, node.indices, node.data, k)
                knn = list(heapq.merge(single_knn, knn))

            # not a leaf
            else:
                # check left branch
                radius = heapq.nlargest(1, knn)
                if self._check_intersection(node.left_hyper_rect, data, radius[0][0]):
                    stack.append(self.tree[node.left])

                # chech right branch
                if self._check_intersection(node.right_hyper_rect, data, radius[0][0]):
                    stack.append(self.tree[node.right]) 
        return knn
    
    def query(self, data, k):
        num_samples, _ = data.shape
        # search kdtree
        knn = []
        for i in np.arange(num_samples):
            single_data = data[i]
            single_knn = self._query_single_data(single_data, k)
            knn.append(single_knn)

        return knn

    def _query_radius_single_data(self, data, radius):
        stack = [self.tree[0]]
        inside = []
        while stack:
            node = stack.pop()
            # leaf
            if node.indices is not None:
                distance = np.sqrt(((node.data - data)**2).sum(axis = 1))
                nn = np.where(distance <= radius)
                if len(nn[0]) > 0:
                    idx = node.indices[nn]
                    distance = distance[nn]
                    inside.extend(list(zip(distance, idx)))
            # branch node
            else:
                if self._check_intersection(node.left_hyper_rect, data, radius):
                    stack.append(self.tree[node.left])

                if self._check_intersection(node.right_hyper_rect, data, radius):
                    stack.append(self.tree[node.right])              
        return inside
    
    def query_radius(self, data, radius):
        num_samples, _ = data.shape
        # search kdtree
        knn = []
        for i in np.arange(num_samples):
            single_data = data[i, :]
            single_knn = self._query_radius_single_data(single_data, radius)
            if single_knn:
                knn.append(single_knn)

        return knn
