from typing import Dict, Callable, Any, Set
import networkx as nx
import decimal


class DIFFUSION:
    def __init__(
            self,
            source: list,
            gamma: float = 0.1,
            epsilon: float = 1e-3,
            grad_func: Callable = lambda _x: _x ** 3,
    ):
        self.graph = nx.DiGraph()
        self.sources = set(source)
        self.gamma = decimal.Decimal(str(gamma))
        self.epsilon = decimal.Decimal(str(epsilon))
        self.grad_func = grad_func

        self.p = {s: decimal.Decimal('0') for s in self.sources}  # 扩散值，初始为0
        self.r = {s: decimal.Decimal('1.0') for s in self.sources}  # 残差初始化为1
        self.q = list(self.sources)
        self.processed_edges = set()

    def edge_arrive(self, u: Any, v: Any, attrs: Dict):
        edge_key = (u, v)
        if edge_key in self.processed_edges:
            return

        self.processed_edges.add(edge_key)

        weight = decimal.Decimal(str(attrs.get('weight', '0')))
        self.graph.add_edge(u, v, weight=weight)

        # 初始化新节点的p值为0（非源节点）
        if u not in self.p:
            self.p[u] = decimal.Decimal('0')
        if v not in self.p:
            self.p[v] = decimal.Decimal('0')
        if u not in self.r:
            self.r[u] = decimal.Decimal('0')
        if v not in self.r:
            self.r[v] = decimal.Decimal('0')

        if (u in self.sources and u not in self.q) or self.r[u] > self.epsilon:
            self.q.append(u)
        if (v in self.sources and v not in self.q) or self.r[v] > self.epsilon:
            self.q.append(v)

        self._process_queue()

    def _process_queue(self):
        while len(self.q) > 0:
            u = self.q.pop(0)
            if self.r[u] <= self.epsilon:
                continue

            dp_max = decimal.Decimal('1') - self.p.get(u, decimal.Decimal('0'))
            dp = dp_max / decimal.Decimal('2')
            is_source = (u in self.sources)

            ru_new = self._eval_residual(u, dp, is_source)
            while ru_new > decimal.Decimal('1') - self.epsilon:
                dp = (dp + dp_max) / decimal.Decimal('2')
                ru_new = self._eval_residual(u, dp, is_source)

            self.r[u] = ru_new

            self._push_to_neighbors(u, dp, is_source)

            old_pu = self.p.get(u, decimal.Decimal('0'))
            self.p[u] = old_pu + dp

    def _eval_residual(self, u: Any, dp: decimal.Decimal, is_source: bool) -> decimal.Decimal:
        if is_source:
            p_u = self.p.get(u, decimal.Decimal('0'))
            return decimal.Decimal(str(self.grad_func(float(p_u + dp - decimal.Decimal('1')))))

        rlt = decimal.Decimal('0')
        out_edges = self.graph.out_edges(u, data=True)
        dist = {v: decimal.Decimal(str(attr.get('weight', '0'))) for _, v, attr in out_edges}
        sum_weight = sum(dist.values())

        if sum_weight == decimal.Decimal('0'):
            return rlt

        for v, w in dist.items():
            p_u = self.p.get(u, decimal.Decimal('0'))
            p_v = self.p.get(v, decimal.Decimal('0'))
            grad_val = decimal.Decimal(str(self.grad_func(float(p_u + dp - p_v))))
            rlt += grad_val * (w / sum_weight)

        return -rlt / self.gamma

    def _push_to_neighbors(self, u: Any, dp: decimal.Decimal, is_source: bool):
        p_u = self.p.get(u, decimal.Decimal('0'))
        new_p_u = p_u + dp

        in_edges = self.graph.in_edges(u, data=True)
        dist = {
            v: decimal.Decimal(str(attr.get('weight', '0')))
            for v, _, attr in in_edges
        }
        sum_weight = sum(dist.values())

        if sum_weight == decimal.Decimal('0'):
            return

        for v, w in dist.items():
            p_v = self.p.get(v, decimal.Decimal('0'))
            delta = decimal.Decimal(str(self.grad_func(float(p_v - p_u)))) - \
                    decimal.Decimal(str(self.grad_func(float(p_v - new_p_u))))

            self.r[v] = self.r.get(v, decimal.Decimal('0')) + (w / sum_weight) * delta / self.gamma

            if self.r[v] > self.epsilon and v not in self.q:
                self.q.append(v)