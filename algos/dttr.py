from typing import List, Dict, Any

import networkx as nx


class DTTR:
    def __init__(
            self, source: List[str],
            alpha: float = 0.15,
            epsilon: float = 1e-3
    ):
        assert 0 <= alpha <= 1
        self.alpha = alpha

        assert 0 < epsilon < 1
        self.epsilon = epsilon

        self.r = {s: 1.0 for s in source}
        self.p = dict()

        self._witness_graph = nx.MultiDiGraph()
        self._witness_graph.add_nodes_from(source)

    def edge_arrive(self, u: Any, v: Any, attrs: Dict):
        self._witness_graph.add_edge(u, v, **attrs)
