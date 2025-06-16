from typing import Any, Dict, List, Set
import networkx as nx
from networkx.algorithms.community import asyn_lpa_communities as lpa

from algos.push_pop import PushPopModel


class LPA(PushPopModel):
    def __init__(self, source: str):
        super().__init__(source)
        self.source = source
        self.visited: Set[str] = set()
        self.processed: Set[str] = set()
        self.community_nodes: Set[str] = {source}
        self.process_queue: List[Dict] = []
        self.queued_nodes: Set[str] = set()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.undirected_graph: nx.Graph = nx.Graph()

        self.edges_added = 0
        self.community_update_count = 0

    def push(self, node: str, edges: List[Dict], **kwargs) -> None:

        if node in self.visited:
            return

        self.visited.add(node)
        self.edges_added = 0

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
            self.edges_added += 1

        # 双阈值控制LPA执行频率
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
        node_id = next_node['node']
        self.processed.add(node_id)
        return next_node

    def _compute_community(self) -> None:
        """
        执行LPA算法更新社区节点
        """
        if not self.undirected_graph.nodes:
            return

        communities = list(lpa(self.undirected_graph, weight='weight'))

        # 找到包含source的社区
        source_community = None
        for comm in communities:
            if self.source in comm:
                source_community = comm
                break

        if source_community:
            self.community_nodes = set(source_community) - self.processed