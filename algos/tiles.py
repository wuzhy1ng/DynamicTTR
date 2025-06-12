import networkx as nx
from networkx.exception import NetworkXError
from queue import PriorityQueue
from typing import List, Dict, Set
import time

class TILES:
    def __init__(
        self,
        source: List[str],
        ttl: float = float('inf'),  # 边过期时间（默认永久有效）
        observation_window: float = 86400  # 观测窗口（默认1天）
    ):
        self.source = source
        self.ttl = ttl
        self.observation_window = observation_window
        self.community_id_counter = 0
        self.digraph = nx.DiGraph()
        self.node_to_communities: Dict[str, Set[int]] = {}
        self.communities: Dict[int, Set[str]] = {}
        self.edge_ttl_queue = PriorityQueue()  # 过期边优先队列
        self.last_observation_time = time.time()  # 最后观测时间

        # 初始化：为每个源节点创建独立社区
        for node in source:
            self._create_new_community([node])

    def _create_new_community(self, initial_nodes: List[str]) -> int:
        """创建新社区"""
        if not initial_nodes:
            return 0
        self.community_id_counter += 1
        self.communities[self.community_id_counter] = set(initial_nodes)
        for node in initial_nodes:
            # 初始化节点社区归属
            self.node_to_communities.setdefault(node, set()).add(self.community_id_counter)
        return self.community_id_counter

    def _is_core_member(self, node: str, community_nodes: Set[str]) -> bool:
        """
        判断节点是否为核心成员（存在有向三角形结构）
        """
        if len(community_nodes) < 3:
            return False

        try:
            neighbors = set(self.digraph.successors(node)) & community_nodes
        except NetworkXError:
            return False

        if len(neighbors) < 2:
            return False  # 至少需要2个邻居才可能形成三角形

        # 取第一个邻居的邻居集合，与剩余邻居求交集（存在交集则形成三角形）
        first_neighbor = next(iter(neighbors))
        first_neighbor_neighbors = set(self.digraph.successors(first_neighbor)) & community_nodes

        # 检查是否有其他邻居在first_neighbor的邻居中
        for neighbor in neighbors:
            if neighbor != first_neighbor and neighbor in first_neighbor_neighbors:
                return True

        return False

    def edge_arrive(self, from_node: str, to_node: str, edge_attrs: Dict):
        timestamp = edge_attrs['timeStamp']

        self.digraph.add_node(from_node)
        self.digraph.add_node(to_node)
        if from_node == to_node:
            return

        self.digraph.add_edge(from_node, to_node, **edge_attrs)
        self.edge_ttl_queue.put((timestamp, from_node, to_node))  # 记录边时间戳

        try:
            from_successors = set(self.digraph.successors(from_node))
            to_successors = set(self.digraph.successors(to_node))
        except NetworkXError:
            return
        common_successors = from_successors & to_successors  # 共同邻居节点

        self._propagate_communities(from_node, to_node, common_successors)
        self._remove_expired_edges(timestamp)
        self._trigger_observation(timestamp)

    def _propagate_communities(self, u: str, v: str, common_neighbors: Set[str]):
        """
        社区标签传播（多米诺效应策略）
        1. 共享社区内的核心节点传播
        2. 独立社区的外围节点扩展
        3. 无共享社区时创建新社区
        """
        u_comms = self.node_to_communities.get(u, set())
        v_comms = self.node_to_communities.get(v, set())
        shared_comms = u_comms & v_comms
        unique_u_comms = u_comms - v_comms
        unique_v_comms = v_comms - u_comms

        # 1. 共享社区内的核心节点传播
        for comm_id in shared_comms:
            comm_nodes = self.communities[comm_id]
            if self._is_core_member(u, comm_nodes) and self._is_core_member(v, comm_nodes):
                # 核心节点传播：将共同邻居加入社区外围
                for neighbor in common_neighbors:
                    self.node_to_communities.setdefault(neighbor, set()).add(comm_id)
                    comm_nodes.add(neighbor)

        # 2. 独立社区的外围节点扩展
        for comm_id in unique_u_comms:
            comm_nodes = self.communities[comm_id]
            if self._is_core_member(u, comm_nodes):
                # 源节点u为核心时，v加入外围
                self.node_to_communities.setdefault(v, set()).add(comm_id)
                comm_nodes.add(v)
        for comm_id in unique_v_comms:
            comm_nodes = self.communities[comm_id]
            if self._is_core_member(v, comm_nodes):
                # 源节点v为核心时，u加入外围
                self.node_to_communities.setdefault(u, set()).add(comm_id)
                comm_nodes.add(u)

        # 3. 无共享社区时创建新社区
        if not shared_comms and common_neighbors:
            new_comm_nodes = {u, v, *common_neighbors}
            core_nodes = {n for n in new_comm_nodes if self._is_core_member(n, new_comm_nodes)}
            if core_nodes:
                comm_id = self._create_new_community(list(core_nodes))
                for node in new_comm_nodes:
                    self.node_to_communities.setdefault(node, set()).add(comm_id)

    def _remove_expired_edges(self, current_time: float):
        """
        移除过期边（时间衰减机制）
        使用TTL参数控制边有效期
        """
        while not self.edge_ttl_queue.empty():
            ts, u, v = self.edge_ttl_queue.queue[0]
            if current_time - ts > self.ttl:
                self.edge_ttl_queue.get()
                if self.digraph.has_edge(u, v):
                    self.digraph.remove_edge(u, v)
                    # 更新社区成员关系
                    shared_comms = self.node_to_communities[u] & self.node_to_communities[v]
                    for comm_id in shared_comms:
                        self.communities[comm_id] -= {u, v}
                        self.node_to_communities[u].discard(comm_id)
                        self.node_to_communities[v].discard(comm_id)
                        if not self.communities[comm_id]:
                            del self.communities[comm_id]
            else:
                break  # 剩余边未过期，停止处理

    def _trigger_observation(self, current_time: float):
        """
        观测窗口触发点
        """
        pass

    @property
    def p(self) -> Dict[str, int]:
        """返回与source节点在同一个社区的所有节点"""
        if not self.source:
            return {}

        # 获取所有source节点所属的社区ID
        source_communities = set()
        for node in self.source:
            if node in self.node_to_communities:
                source_communities.update(self.node_to_communities[node])

        # 若source节点没有所属社区，返回空字典
        if not source_communities:
            return {}

        # 收集这些社区中的所有节点
        same_community_nodes = set()
        for comm_id in source_communities:
            if comm_id in self.communities:
                same_community_nodes.update(self.communities[comm_id])

        return {node: 1 for node in same_community_nodes}
