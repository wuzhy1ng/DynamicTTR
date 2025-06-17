import decimal
import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Dict, Any, Set

import networkx as nx

from utils.price import get_usd_value

_NUM_ZERO = decimal.Decimal('0')
_NUM_ONE = decimal.Decimal('1')
_NUM_TWO = decimal.Decimal('2')


class DTTR:
    def __init__(
            self, source: List[str],
            alpha: float = 0.15,
            epsilon: float = 0.001,
            is_in_usd: bool = True,
            is_reduce_swap: bool = True,
            is_log_value: bool = True,
    ):
        assert 0 <= alpha <= 1
        self.alpha = decimal.Decimal(alpha)
        assert 0 < epsilon < 1
        self.epsilon = decimal.Decimal(epsilon)
        self.is_in_usd = is_in_usd
        self.is_reduce_swap = is_reduce_swap
        self.is_log_value = is_log_value

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
        # self._edge_weights = list()

        self._pool = ThreadPoolExecutor(max_workers=max(os.cpu_count() // 2, 1))

    def transaction_arrive(self, g: nx.MultiDiGraph):
        # filter out the zero value edges
        # and the trans. without moving the mass from supporting nodes
        edges = [
            (u, v, attr) for u, v, attr in g.edges(data=True)
            if float(attr.get('value', '0')) != 0 and any([
                self.p.get(u), self.p.get(v)
            ])
        ]
        if len(edges) == 0:
            return

        # transform the value to usd price
        if self.is_in_usd:
            params = [
                (attr['contractAddress'], attr['value'], attr['timeStamp'])
                for _, _, attr in edges
            ]
            results = self._pool.map(
                lambda args: get_usd_value(*args),
                params
            )
            nonzero_edges = list()
            for edge, result in zip(edges, results):
                u, v, attr = edge
                if result == '0':
                    continue
                attr['value'] = result
                nonzero_edges.append(edge)
            edges = nonzero_edges

        # model as a graph and delete the swaps
        for _, _, attr in edges:
            attr['value'] = decimal.Decimal(attr['value'])
        nonzero_g = nx.MultiDiGraph()
        nonzero_g.add_edges_from(edges)
        if self.is_reduce_swap:
            nodes = self._det_swap_nodes(nonzero_g)
            nonzero_g.remove_nodes_from(nodes)

        # update mass and local push
        for u, v, attr in nonzero_g.edges(data=True):
            if not self.p.get(u):
                continue

            # add restart edges
            if not self._node2outsum.get(v):
                for s in self.source:
                    self._witness_graph.add_edge(v, s, value=self.epsilon)
                self._node2outsum[v] = self.epsilon * len(self.source)
                # self._witness_graph.add_edge(v, v, value=self.epsilon)
                # self._node2outsum[v] = self.epsilon

            # update mass and local push
            nodes_push = self._update_mass(u, v, attr['value'])
            self._local_push(nodes_push)

    def _det_swap_nodes(self, g: nx.MultiDiGraph) -> Set:
        result = set()
        node2outcome, node2income = dict(), dict()
        for u, v, attr in g.edges(data=True):
            node2outcome[u] = node2outcome.get(u, _NUM_ZERO) + attr['value']
            node2income[v] = node2income.get(v, _NUM_ZERO) + attr['value']
        for node in [node for node in g.nodes()]:
            outcome = node2outcome.get(node, _NUM_ZERO)
            income = node2income.get(node, _NUM_ZERO)
            if outcome.is_zero() or income.is_zero():
                continue
            rate = 1 - min(income, outcome) / max(outcome, income)
            if rate <= 0.05:
                result.add(node)
        return result

    def _update_mass(self, u: str, v: str, value: decimal.Decimal) -> Set:
        if self.is_log_value:
            value = value + _NUM_ONE
            value = value.log10()
            # self._edge_weights.append(float(value))

        # init args
        d_out_old = self._node2outsum[u]
        d_out_new = d_out_old + value
        self._node2outsum[u] = d_out_new
        nodes_push = set()

        # update node u
        pu_old = self.p[u]
        self.p[u] *= d_out_new / d_out_old
        delta_ru = - (1 / self.alpha) * pu_old
        delta_ru *= value / d_out_old
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
            self._witness_graph.add_edge(u, v, value=value)
        else:
            edge_data['value'] += value
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


if __name__ == '__main__':
    model = DTTR(
        source=['a'],
        alpha=0.15,
        epsilon=1e-3,
        is_in_usd=False,
    )
    trans = nx.MultiDiGraph()
    trans.add_edges_from([
        ('a', 'b', {'value': '10'}),
        ('b', 'a', {'value': '9.8'}),
        ('a', 'c', {'value': '0.2'}),
    ])
    model.transaction_arrive(trans)
    print(model.p, model.r)

    trans = nx.MultiDiGraph()
    trans.add_edges_from([
        ('a', 'c', {'value': '0.1'}),
    ])
    model.transaction_arrive(trans)
    print(model.p, model.r)

    trans = nx.MultiDiGraph()
    trans.add_edges_from([
        ('a', 'd', {'value': '9.5'}),
    ])
    model.transaction_arrive(trans)
    print(model.p, model.r)

    trans = nx.MultiDiGraph()
    trans.add_edges_from([
        ('d', 'e', {'value': '9'}),
    ])
    model.transaction_arrive(trans)
    print(model.p, model.r)

    # trans = nx.MultiDiGraph()
    # trans.add_edges_from([
    #     ('a', 'e', {'value': '9'}),
    # ])
    # model.transaction_arrive(trans)
    # model.edge_arrive('a', 'b', {'value': 10})
    # model.edge_arrive('b', 'a', {'value': 9.8})
    # model.edge_arrive('a', 'c', {'value': 9.7})
    # model.edge_arrive('c', 'd', {'value': 9})
    # print(model.p, model.r)
