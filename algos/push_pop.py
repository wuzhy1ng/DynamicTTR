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
            for edge_view in [graph.in_edges(src, data=True), graph.out_edges(src, data=True)]:
                for u, v, attrs in edge_view:
                    edges.append({'from': u, 'to': v, **attrs})
            self.push(self.source, edges)

        # expanding
        node = self.pop()
        vis = {node['node']} if isinstance(node, dict) else {node}

        while node is not None:
            edges = list()
            for edge_view in [graph.in_edges(node, data=True), graph.out_edges(node, data=True)]:
                for u, v, attrs in edge_view:
                    edges.append({'from': u, 'to': v, **attrs})
            self.push(node, edges)

            node = self.pop()  # 重新获取新节点
            if node is not None:
                node = node['node'] if isinstance(node, dict) else node
                vis.add(node)

        return graph.subgraph(list(vis))
