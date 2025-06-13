from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import community.community_louvain as community_louvain

from algos.push_pop import PushPopModel


class LOUVAIN(PushPopModel):
    def __init__(self, source: str):
        super().__init__(source)
        self.source = source
        self.visited: Set[str] = set()
        self.community_nodes: Set[str] = set()
        self.process_queue: List[Dict] = []
        self.current_community: int = None
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.undirected_graph: nx.Graph = nx.Graph()

    def push(self, node: str, edges: List[Dict], **kwargs) -> None:

        self.visited.add(node)

        for edge in edges:
            u = edge['from']
            v = edge['to']
            value = float(edge.get('value', 0))

            if u == v:
                continue

            self.graph.add_edge(u, v, weight=value)

            if self.undirected_graph.has_edge(u, v):
                current_weight = self.undirected_graph[u][v].get('weight', 0)
                self.undirected_graph[u][v]['weight'] = current_weight + value
            else:
                self.undirected_graph.add_edge(u, v, weight=value)

        if node == self.source or not self.community_nodes:
            self._compute_community()

            self.process_queue = [
                {'node': n} for n in self.community_nodes
                if n not in self.visited and n != self.source
            ]

    def pop(self) -> Dict[str, Any]:
        if not self.process_queue:
            return None

        next_node_info = self.process_queue.pop(0)
        return next_node_info

    def _compute_community(self) -> None:
        if not self.undirected_graph.nodes:
            return

        partition = community_louvain.best_partition(
            self.undirected_graph,
            weight='weight',
            resolution=1.0,
            random_state=42
        )

        # 确定源节点所在的社区
        source_community = partition.get(self.source)
        if source_community is not None:
            # 收集属于源节点所在的社区的所有节点
            self.community_nodes = {
                node for node, comm in partition.items()
                if comm == source_community
            }
            self.current_community = source_community
