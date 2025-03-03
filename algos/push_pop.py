from typing import Any

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


class PushPopAggregator:
    def __init__(
            self, source_list: list,
            model_cls: Any,
    ):
        self.source_list = source_list
        self.model_cls = model_cls

    def execute(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        vis = set(self.source_list)
        for src in self.source_list:
            # get the instance
            model = self.model_cls(src)

            # init on the source
            edges = list()
            for edge_view in [
                graph.in_edges(src, data=True),
                graph.out_edges(src, data=True)
            ]:
                for u, v, attrs in edge_view:
                    edges.append({'from': u, 'to': v, **attrs})
            model.push(src, edges)

            # expanding
            node = model.pop()
            node = node['node'] if isinstance(node, dict) else node
            while node is not None:
                edges = list()
                for edge_view in [
                    graph.in_edges(node, data=True),
                    graph.out_edges(node, data=True)
                ]:
                    for u, v, attrs in edge_view:
                        edges.append({'from': u, 'to': v, **attrs})
                model.push(node, edges)

                # get the next node
                node = model.pop()
                if node is not None:
                    node = node['node'] if isinstance(node, dict) else node
                    vis.add(node)

        return graph.subgraph(list(vis))
