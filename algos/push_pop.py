import networkx as nx


class PushPopModel:
    def __init__(self, source):
        self.source = source

    def push(self, node, edges: list, **kwargs):
        """
        push a node with related edges
        :param node:
        :param edges:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def pop(self):
        """
        pop a series of nodes
        :return:
        """
        raise NotImplementedError()

    def execute(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        # init on the source
        edges = list()
        for src in self.source:
            for edge_view in [graph.in_edges(src), graph.out_edges(src)]:
                for u, v, attrs in edge_view:
                    edges.append({'from': u, 'to': v, **attrs})
            _ = [item for item in self.push(self.src, edges)]

        # expanding
        node = self.pop()
        vis = {node}
        while node is not None:
            edges = list()
            for edge_view in [graph.in_edges(node), graph.out_edges(node)]:
                for u, v, attrs in edge_view:
                    edges.append({'from': u, 'to': v, **attrs})
            _ = [item for item in self.push(node, edges)]
            node = node.pop()
            vis.add(node)
        return graph.subgraph(list(vis))
