from queue import Queue

from algos.push_pop import PushPopModel


class BFS(PushPopModel):
    def __init__(self, source, depth: int = 2):
        super().__init__(source)
        self.depth = depth
        self._vis = {self.source}
        self._queue = Queue()

    def push(self, node, edges: list, depth: int = 0):
        """
        push a node with related edges, and the edges requires `from` and `to`
        :param node:
        :param edges:
        :param depth:
        :return:
        """
        assert depth >= 0

        if depth + 1 > self.depth:
            return

        for e in edges:
            self._queue.put((e.get('from'), depth + 1))
            self._queue.put((e.get('to'), depth + 1))

    def pop(self):
        while not self._queue.empty():
            node, depth = self._queue.get()
            if node not in self._vis and depth <= self.depth:
                self._vis.add(node)
                return dict(node=node, depth=depth)
        return None
