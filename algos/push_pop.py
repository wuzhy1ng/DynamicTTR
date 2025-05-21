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
            self, source: Any,
            model_cls: Any,
    ):
        self.source = source
        self.model_cls = model_cls

    def execute(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        src = self.source
        vis = {self.source}

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
        pop_item = model.pop()
        node = pop_item['node'] if isinstance(pop_item, dict) else None
        while pop_item is not None:
            edges = list()
            for edge_view in [
                graph.in_edges(node, data=True),
                graph.out_edges(node, data=True)
            ]:
                for u, v, attrs in edge_view:
                    edges.append({'from': u, 'to': v, **attrs})
            pop_item.pop('node')
            model.push(node, edges, **pop_item)

            # get the next node
            pop_item = model.pop()
            if pop_item is not None:
                node = pop_item['node'] if isinstance(pop_item, dict) else None
                vis.add(node)

        return graph.subgraph(list(vis))
