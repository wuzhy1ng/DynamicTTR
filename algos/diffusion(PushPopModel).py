from typing import Dict, Callable, Any
import networkx as nx
from algos.push_pop import PushPopModel


class DIFFUSION(PushPopModel):
    def __init__(self, source, gamma: float = 0.1, epsilon: float = 1e-3,
                 grad_func: Callable = lambda _x: _x ** 3):
        super().__init__(source)
        self.gamma = gamma
        self.epsilon = epsilon
        self.grad_func = grad_func
        self.p = {source: 0.0}
        self.r = {source: 1.0}
        self.queue = [source]

    def push(self, node, edges: list, **kwargs):
        if self.r.get(node, 0) <= self.epsilon:
            return

        p_max = 1 - self.p.get(node, 0)
        dp = p_max / 2
        is_source = (node == self.source)

        ru_new = self._eval_residual(node, dp, is_source, edges)

        while ru_new > 1 - self.epsilon:
            dp = (dp + p_max) / 2
            ru_new = self._eval_residual(node, dp, is_source, edges)

        self.r[node] = ru_new

        in_edges = [e for e in edges if e.get('to') == node]
        predecessors = {}

        for e in in_edges:
            u = e.get('from')
            weight = e.get('weight', 0)
            predecessors[u] = predecessors.get(u, 0) + weight

        sum_weight = sum(predecessors.values())

        for neighbor, weight in predecessors.items():
            if sum_weight == 0:
                weight = 1.0 / len(predecessors) if predecessors else 0
            else:
                weight /= sum_weight

            delta = self.grad_func(self.p.get(neighbor, 0) - self.p.get(node, 0)) - \
                    self.grad_func(self.p.get(neighbor, 0) - (self.p.get(node, 0) + dp))

            self.r[neighbor] = self.r.get(neighbor, 0) + weight * delta / self.gamma

            if self.r[neighbor] > self.epsilon and neighbor not in self.p:
                self.queue.append(neighbor)
                self.p[neighbor] = 0.0

        self.p[node] = self.p.get(node, 0) + dp

    def pop(self):
        if not self.queue:
            return None
        return {'node': self.queue.pop(0)}

    def _eval_residual(self, node, dp, is_source, edges):
        if is_source:
            return self.grad_func(self.p.get(node, 0) + dp - 1)

        out_edges = [e for e in edges if e.get('from') == node]
        successors = {}

        for e in out_edges:
            v = e.get('to')
            weight = e.get('weight', 0)
            successors[v] = successors.get(v, 0) + weight

        sum_weight = sum(successors.values())

        rlt = 0
        for neighbor, weight in successors.items():
            if sum_weight > 0:
                weight /= sum_weight
            else:
                weight = 1.0 / len(successors) if successors else 0

            rlt += self.grad_func(self.p.get(node, 0) + dp - self.p.get(neighbor, 0)) * weight

        return -rlt / self.gamma