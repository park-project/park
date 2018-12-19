import numpy as np
from park import core
from park.spaces.rng import np_random, nrng
from park.utils.directed_graph import DirectedGraph


class EdgeInGraph(core.Space):
    """
    The element of this space is an edge in a DirectedGraph object.
    """
    def __init__(self, graph=None):
        if graph is not None:
            assert type(graph) == DirectedGraph
        self.graph = graph
        core.Space.__init__(self, 'graph_float32', (), np.float32)

    def update_graph(graph):
        assert type(graph) == DirectedGraph
        self.graph = graph

    def sample(self, valid_list=None):
        if valid_list is None:
            return nrng.choice(self.graph.edges)
        else:
            assert len(valid_list) <= len(self.graph.edges)
            return valid_list[np_random.randint(len(valid_list))]

    def contains(self, x):
        return self.graph.has_edge(x)
