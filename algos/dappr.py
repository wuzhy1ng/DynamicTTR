import decimal
import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Dict, Any, Set

import networkx as nx

_NUM_ZERO = decimal.Decimal('0')
_NUM_ONE = decimal.Decimal('1')
_NUM_TWO = decimal.Decimal('2')


class DAPPR:
    def __init__(
            self, source: List[str],
            alpha: float = 0.15,
            epsilon: float = 1e-3,
            is_in_usd: bool = True,
    ):
        assert 0 <= alpha <= 1
        self.alpha = decimal.Decimal(alpha)
        assert 0 < epsilon < 1
        self.epsilon = decimal.Decimal(epsilon)
        self.is_in_usd = is_in_usd

        self.r = dict()
        self.p = {s: _NUM_ONE for s in source}
        self.source = source

        self._node2outsum = {s: self.epsilon for s in self.source}
        self._witness_graph = nx.DiGraph()
        self._witness_graph.add_nodes_from(source)
        self._witness_graph.add_edges_from([
            (s, s, {'value': self.epsilon})
            for s in self.source
        ])

        self._pool = ThreadPoolExecutor(max_workers=max(os.cpu_count() // 2, 1))

    def edge_arrive(self, u: Any, v: Any, attrs: Dict):
        """
        Update the witness graph if edge arrived.

        :param u: the node from
        :param v: the node to
        :param attrs: the edge attributes, MUST include `value`
        :return:
        """
        if not self.p.get(u):
            return

        # add restart edges
        if not self._node2outsum.get(v):
            self._witness_graph.add_edge(v, v, value=self.epsilon)
            self._node2outsum[v] = self.epsilon

        # update mass and local push
        nodes_push = self._update_mass(u, v)
        self._local_push(nodes_push)

    def _update_mass(self, u: str, v: str) -> Set:
        # init args
        d_out_old = self._node2outsum[u]
        d_out_new = d_out_old + 1
        self._node2outsum[u] = d_out_new
        nodes_push = set()

        # update node u
        pu_old = self.p.get(u, 0)
        self.p[u] *= d_out_new / d_out_old
        delta_ru = - (1 / self.alpha) * pu_old
        delta_ru *= 1 / d_out_old
        self.r[u] = self.r.get(u, _NUM_ZERO) + delta_ru
        if abs(self.r[u]) >= self.epsilon:
            nodes_push.add(u)

        # update node v
        self.r[v] = self.r.get(v, _NUM_ZERO) - delta_ru * (1 - self.alpha)
        if abs(self.r[v]) >= self.epsilon:
            nodes_push.add(v)

        # record the new edge
        edge_data = self._witness_graph.get_edge_data(u, v)
        if edge_data is None:
            self._witness_graph.add_edge(u, v, value=1)
        else:
            edge_data['value'] += 1
        return nodes_push

    def _local_push(self, nodes_push: Set[Any]):
        while len(nodes_push) > 0:
            node_residual = [(node, self.r[node]) for node in nodes_push]
            node, residual = max(node_residual, key=lambda x: abs(x[1]))
            nodes_push.remove(node)
            self.r[node] = _NUM_ZERO
            self.p[node] = self.p.get(node, _NUM_ZERO) + residual * self.alpha
            outsum = self._node2outsum[node]
            out_edges = [
                (neighbor, _attrs)
                for _, neighbor, _attrs
                in self._witness_graph.out_edges(node, data=True)
            ]
            if len(out_edges) == 1 and out_edges[0][0] == node:
                self.p[node] += residual * (1 - self.alpha)
                continue
            for neighbor, _attrs in out_edges:
                _value = decimal.Decimal(_attrs['value'])
                fac = _value / outsum
                self.r[neighbor] = self.r.get(neighbor, _NUM_ZERO) + residual * (1 - self.alpha) * fac
                if abs(self.r[neighbor]) >= self.epsilon:
                    nodes_push.add(neighbor)
