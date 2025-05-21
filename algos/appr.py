from algos.push_pop import PushPopModel
from utils.cache import LRUCache


class APPR(PushPopModel):
    def __init__(self, source, alpha: float = 0.15, epsilon: float = 1e-3):
        super().__init__(source)

        assert 0 <= alpha <= 1
        self.alpha = alpha

        assert 0 < epsilon < 1
        self.epsilon = epsilon

        self.r = {self.source: 1}
        self.p = dict()

        self._cache = LRUCache()

    def push(self, node, edges: list, **kwargs):
        r_node = self.r.get(node, 0)
        if r_node == 0:
            return
        self.r[node] = 0
        self.p[node] = self.p.get(node, 0) + r_node * self.alpha

        neighbours = set()
        for e in edges:
            neighbours.add(e.get('from'))
            neighbours.add(e.get('to'))
        if node in neighbours:
            neighbours.remove(node)

        neighbours_cnt = len(neighbours)
        inc = (1 - self.alpha) * r_node / neighbours_cnt if neighbours_cnt > 0 else 0
        for neighbour in neighbours:
            self.r[neighbour] = self.r.get(neighbour, 0) + inc

    def pop(self):
        while True:
            node, r_node = None, None
            for _node, _r_node in self.r.items():
                if _r_node <= self.epsilon or not self._cache.get(_node):
                    continue
                node, r_node = _node, _r_node
                break

            if not node:
                break
            self.r[node] = 0
            self.p[node] = self.p.get(node, 0) + r_node * self.alpha
            for v, d in self._cache.get(node).items():
                self.r[v] = self.r.get(v, 0) + d * r_node

        node, r = None, self.epsilon
        for _node, _r in self.r.items():
            if _r > r:
                node, r = _node, _r

        return dict(node=node, residual=r) if node is not None else None
