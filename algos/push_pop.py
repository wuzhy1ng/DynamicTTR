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
        vis = self.source
        for s in self.source:
            edges = list()
            for edge_view in [graph.in_edges(s, data=True), graph.out_edges(s, data=True)]:
                for u, v, attrs in edge_view:
                    edges.append({'from': u, 'to': v, **attrs})
            _ = [self.push(s, edges)]

        # expanding
        node = self.pop()
        while node is not None:
            if node['node'] not in vis:
                edges = list()
                for edge_view in [graph.in_edges(node['node'], data=True), graph.out_edges(node['node'], data=True)]:
                    for u, v, attrs in edge_view:
                        edges.append({'from': u, 'to': v, **attrs})
                _ = [self.push(node['node'], edges)]
                vis.add(node['node'])
            node = self.pop()
        return graph.subgraph(list(vis))
