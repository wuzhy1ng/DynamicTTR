import decimal
from typing import List, Dict, Any, Set

import networkx as nx

from utils.price import get_usd_price

_NUM_ZERO = decimal.Decimal('0')
_NUM_ONE = decimal.Decimal('1')


class DTTR:
    def __init__(
            self, source: List[str],
            alpha: float = 0.15,
            epsilon: float = 1e-4,
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

    def edge_arrive(self, u: Any, v: Any, attrs: Dict):
        """
        Update the witness graph if edge arrived.

        :param u: the node from
        :param v: the node to
        :param attrs: the edge attributes, MUST include `value`
        :return:
        """
        # filter out the edge
        # if there is no mass from u or the value is zero
        # see the invariant for details
        value = attrs.get('value', '0')
        if not self.p.get(u) or value == '0':
            return
        value = decimal.Decimal(value)
        if self.is_in_usd:
            value = self._value2usd(
                contract_address=attrs['contractAddress'],
                value=value,
                timestamp=attrs['timeStamp'],
            )
        if value.is_zero():  # double check whether the usd price is zero
            return

        # add restart edges
        if not self._node2outsum.get(v):
            self._witness_graph.add_edge(v, v, value=self.epsilon)
            self._node2outsum[v] = self.epsilon

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

        # record the new edge and run a local push
        edge_data = self._witness_graph.get_edge_data(u, v)
        if edge_data is None:
            self._witness_graph.add_edge(u, v, value=value)
        else:
            edge_data['value'] += value
        self._local_push(nodes_push)

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

    def _value2usd(
            self, contract_address: str,
            value: decimal.Decimal,
            timestamp: int
    ) -> decimal.Decimal:
        data = get_usd_price(
            contract_address=contract_address,
            timestamp=timestamp
        )
        if data is None:
            return _NUM_ZERO
        token_decimals = decimal.Decimal(data['decimals'])
        value = value / (decimal.Decimal('10') ** token_decimals)
        price = decimal.Decimal(data['price'])
        return value * price


if __name__ == '__main__':
    model = DTTR(
        source=['a'],
        alpha=0.15,
        epsilon=1e-3,
        is_in_usd=False,
    )
    model.edge_arrive('a', 'b', {'value': 1})
    model.edge_arrive('b', 'c', {'value': 2e18})
    model.edge_arrive('b', 'a', {'value': 1})
    model.edge_arrive('b', 'd', {'value': 1})
    print(model.p, model.r)
