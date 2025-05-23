from typing import Any, Dict, List
from queue import Queue
from algos.push_pop import PushPopModel


class TPP(PushPopModel):
    def __init__(self, source: str, max_depth: int = 20,
                 transaction_threshold: int = 1000,
                 min_amount_threshold: float = 0.01):
        super().__init__(source)
        self.max_depth = max_depth
        self.transaction_threshold = transaction_threshold
        self.min_amount_threshold = min_amount_threshold
        self._vis = {self.source}
        self._queue = Queue()
        self._depth_map = {self.source: 0}
        self._transaction_count = {}

        self._queue.put((source, 0))

    def push(self, node: str, edges: List[Dict[str, Any]], depth: int = 0):

        if depth >= self.max_depth:
            return

        self._transaction_count[node] = self._transaction_count.get(node, 0) + len(edges)

        if self._transaction_count[node] > self.transaction_threshold:
            return

        for edge in edges:
            to_node = edge.get('to')
            if to_node in self._vis:
                continue

            value = float(edge.get('value', 0))
            if value <= self.min_amount_threshold:
                continue

            self._transaction_count[to_node] = self._transaction_count.get(to_node, 0) + 1

            if self._transaction_count.get(to_node, 0) > self.transaction_threshold:
                continue

            self._queue.put((to_node, depth + 1))
            self._depth_map[to_node] = depth + 1

    def pop(self) -> Dict[str, Any]:

        while not self._queue.empty():
            node, depth = self._queue.get()
            if node not in self._vis:
                self._vis.add(node)
                return {'node': node, 'depth': depth}
        return None