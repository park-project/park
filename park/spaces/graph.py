import numpy as np
from park import core
from park.spaces.rng import np_random


class Graph(core.Space):
    """
    The element of this space is a tuple
    [n * m tensor, n * n sparse matrix]
    m is number of features associated with each node of the graph(s),
    n number of nodes, which can be a variable number at each MDP step.
    """
    def __init__(self, low=None, high=None):
        assert low.shape == high.shape
        assert len(low.shape) == 1
        self.low = low
        self.high = high
        self.m = low.shape[0]
        self.n = None
        core.Space.__init__(self, 'graph_float32', (), np.float32)

    def update(self, n):
        self.n = n

    def sample(self, valid_list=None):
        if valid_list is None:
            return np_random.randint(self.n)
        else:
            assert len(valid_list) <= self.n
            return valid_list[np_random.randint(len(valid_list))]

    def contains(self, x):
        node_features, adjacency_mat = x
        return len(node_features.shape) == 2 and \
               node_features.shape[0] == self.n and \
               node_features.shape[1] == self.m and \
               len(adjacency_mat.shape) == 2 and \
               adjacency_mat.shape[0] == adjacency_mat.shape[1] and \
               adjacency_mat.shape[0] == self.n \
               and (node_features >= self.low).all() \
               and (node_features <= self.high).all()
