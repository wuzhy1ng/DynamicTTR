from typing import List, Dict, Any, Set

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

        self.r = dict()
        self.p = {s: 1.0 for s in source}

        self._node2outsum = dict()
        self._witness_graph = nx.DiGraph()
        self._witness_graph.add_nodes_from(source)

    def edge_arrive(self, u: Any, v: Any, attrs: Dict):
        """
        Update the witness graph if edge arrived.

        :param u: the node from
        :param v: the node to
        :param attrs: the edge attributes, MUST include `value`
        :return:
        """
        # filter out the edge
        # if there is no mass from u
        # see the invariant for details
        if not self._has_mass(u):
            return

            # init args
        d_out_old = self._node2outsum.get(u, 0)
        d_out_new = d_out_old + attrs.get('value', 0)
        nodes_push = set()

        # update the residual for out-degree neighbors
        for _, k, _attrs in self._witness_graph.out_edges(u, data=True):
            delta = _attrs.get('value', 0) * self.p.get(u, 0) / self.alpha
            delta *= (1 / d_out_new - 1 / d_out_old)
            self.r[k] = self.r.get(k, 0) + delta
            if self.r[k] >= self.epsilon:
                nodes_push.add(k)

        # update the residual for node v
        value_old = self._witness_graph.get_edge_data(u, v)
        value_old = value_old.get('value', 0) if value_old else 0
        value_new = value_old + attrs.get('value', 0)
        delta = self.p.get(u, 0) / self.alpha
        delta *= ((value_new / d_out_new - value_old / d_out_old)
                  if value_old > 0 else (value_new / d_out_new))
        self.r[v] = self.r.get(v, 0) + delta
        if self.r[v] >= self.epsilon:
            nodes_push.add(v)

        # run a local push
        self._local_push(nodes_push)

    def _has_mass(self, u: Any) -> bool:
        return self.p.get(u, 0) > 0

    def _local_push(self, nodes_push: Set[Any]):
        while len(nodes_push) > 0:
            node = nodes_push.pop()
            residual, self.r[node] = self.r[node], 0
            outsum = self._node2outsum[node]
            self.p[node] = residual * self.alpha
            for _, neighbor, _attrs in self._witness_graph.out_edges(node, data=True):
                fac = _attrs.get('value', 0) / outsum
                self.r[neighbor] += residual * (1 - self.alpha) * fac
                if self.r[neighbor] >= self.epsilon:
                    nodes_push.add(neighbor)


if __name__ == '__main__':
    model = DTTR(
        source=['a'],
        alpha=0.15,
        epsilon=1e-3,
    )
    model.edge_arrive('a', 'b', {'value': 1})
    model.edge_arrive('a', 'c', {'value': 2})
    model.edge_arrive('c', 'b', {'value': 2})
    model.edge_arrive('_a', 'a', {'value': 1})
    print(model.p, model.r)
