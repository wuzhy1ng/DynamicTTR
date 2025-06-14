from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import community.community_louvain as community_louvain

from algos.push_pop import PushPopModel


class LOUVAIN(PushPopModel):
    def __init__(self, source: str):
        super().__init__(source)
        self.source = source
        self.visited: Set[str] = set()
        self.processed: Set[str] = set()
        self.community_nodes: Set[str] = set()
        self.process_queue: List[Dict] = []
        self.queued_nodes: Set[str] = set()
        self.current_community: int = None
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.undirected_graph: nx.Graph = nx.Graph()
        # 新增双阈值控制变量
        self.edges_added = 0           # push新增的边数
        self.community_update_count = 0  # 累计需要push次数

    def push(self, node: str, edges: List[Dict], **kwargs) -> None:

        if node in self.visited:
            return

        self.visited.add(node)
        self.edges_added = 0  # 重置本次新增边数计数器

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
            self.edges_added += 1  # 统计新增边数

        if self.edges_added >= 5 or self.community_update_count >= 10:
            self._compute_community()
            self.community_update_count = 0  # 重置累计次数
        else:
            self.community_update_count += 1  # 增加累计次数

        new_nodes = [
            {'node': n} for n in self.community_nodes
            if n not in self.processed and
               n != self.source and
               n not in self.queued_nodes
        ]

        self.process_queue.extend(new_nodes)
        self.queued_nodes.update({item['node'] for item in self.process_queue})

    def pop(self) -> Dict[str, Any]:
        if not self.process_queue:
            return None

        next_node = self.process_queue.pop(0)
        self.processed.add(next_node['node'])
        return next_node

    def _compute_community(self) -> None:
        if not self.undirected_graph.nodes:
            return

        partition = community_louvain.best_partition(
            self.undirected_graph,
            weight='weight',
            resolution=1.0,
            random_state=42
        )

        source_community = partition.get(self.source)
        if source_community is not None:
            self.community_nodes = {
                node for node, comm in partition.items()
                if comm == source_community
            }
            self.current_community = source_community
