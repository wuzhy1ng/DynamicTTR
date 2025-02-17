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
        self.source = source

        self._node2outsum = {
            s: self.epsilon
            for s in self.source
        }
        self._witness_graph = nx.DiGraph()
        self._witness_graph.add_nodes_from(source)
        self._witness_graph.add_edges_from([
            (s, s, {'value': self.epsilon})
            for s in self.source
        ])

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
        if self.p.get(u, 0) == 0:
            return

        # add restart edges
        if len(self._witness_graph.out_edges(v)) == 0:
            self._witness_graph.add_edges_from([
                (v, s, {'value': self.epsilon})
                for s in self.source
            ])
            self._node2outsum[v] = self.epsilon * len(self.source)

        # init args
        d_out_old = self._node2outsum[u]
        d_out_new = d_out_old + attrs.get('value', 0)
        self._node2outsum[u] = d_out_new
        nodes_push = set()

        # update node u
        pu_old = self.p[u]
        self.p[u] *= (d_out_new / d_out_old) if d_out_old > 0 else 1
        delta_ru = - (1 / self.alpha) * pu_old
        delta_ru *= (attrs.get('value', 0) / d_out_old) if d_out_old > 0 else 1
        self.r[u] = self.r.get(u, 0) + delta_ru
        if abs(self.r[u]) >= self.epsilon:
            nodes_push.add(u)

        # update node v
        self.r[v] = self.r.get(v, 0) - delta_ru * (1 - self.alpha)
        if abs(self.r[v]) >= self.epsilon:
            nodes_push.add(v)

        # record the new edge and run a local push
        edge_data = self._witness_graph.get_edge_data(u, v)
        if edge_data is None:
            self._witness_graph.add_edge(u, v, value=attrs.get('value', 0))
        else:
            edge_data['value'] += attrs.get('value', 0)
        self._local_push(nodes_push)

    def _local_push(self, nodes_push: Set[Any]):
        while len(nodes_push) > 0:
            node = nodes_push.pop()
            residual, self.r[node] = self.r[node], 0
            self.p[node] = self.p.get(node, 0) + residual * self.alpha
            outsum = self._node2outsum[node]
            for _, neighbor, _attrs in self._witness_graph.out_edges(node, data=True):
                fac = _attrs.get('value', 0) / outsum
                self.r[neighbor] += residual * (1 - self.alpha) * fac
                if abs(self.r[neighbor]) >= self.epsilon:
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
