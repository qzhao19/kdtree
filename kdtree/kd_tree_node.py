#!/bin/env python

class KDTreeNode(object):
    def __init__(self, left, right, data, indices, left_hyper_rect, right_hyper_rect):
        self.left = left
        self.right = right
        self.data = data
        self.indices = indices
        self.left_hyper_rect = left_hyper_rect
        self.right_hyper_rect = right_hyper_rect

