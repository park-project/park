import numpy as np
from park import core
from park.spaces.rng import np_random
from park.utils.directed_graph import DirectedGraph


class Graph(core.Space):
    """
    The element of this space is a DirectedGraph object.
    The node features and edge features need to be confined
    within their ranges.
    """
    def __init__(self, node_low=None, node_high=None, edge_low=None, edge_high=None):
        if node_low is not None or node_high is not None:
            assert node_low.shape == node_high.shape
            assert len(node_low.shape) == 1
        if edge_low is not None or edge_high is not None:
            assert edge_low.shape == edge_high.shape
            assert len(edge_low.shape) == 1
        self.node_low = node_low
        self.node_high = node_high
        self.edge_low = edge_low
        self.edge_high = edge_high
        core.Space.__init__(self, 'graph_float32', (), np.float32)

    def sample(self, valid_list=None):
        if valid_list is None:
            return np_random.randint(self.n)
        else:
            assert len(valid_list) <= self.n
            return valid_list[np_random.randint(len(valid_list))]

    def contains(self, x):
        is_element = type(x) == DirectedGraph
        if is_element:
            # Note: this step can be slow
            node_features, _ = x.get_node_features_tensor()
            edge_features, _ = x.get_edge_features_tensor()

            node_check = self.node_low is None or \
                         self.node_high is None or \
                         len(node_features) == 0 or \
                         ((node_features >= self.node_low).all() and \
                         (node_features <= self.node_high).all())

            edge_check = self.edge_low is None or \
                         self.edge_high is None or \
                         len(edge_features) == 0 or \
                         ((edge_features >= self.edge_low).all() and \
                         (edge_features <= self.edge_high).all())

            is_element = node_check and edge_check

        return is_element