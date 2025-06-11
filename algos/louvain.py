from typing import Any, Dict, List, Tuple, Set, DefaultDict
import networkx as nx
from queue import Queue
import math
from collections import defaultdict
from tqdm import tqdm

from algos.push_pop import PushPopModel


class LOUVAIN(PushPopModel):
    def __init__(self, source: str):
        super().__init__(source)
        self.current_community: Dict[str, str] = {}  # 节点到社区的映射
        self.community_sum_in: Dict[str, float] = {}  # 社区内有向边权重和（u→v且u,v同社区）
        self.community_sum_out: Dict[str, float] = {}  # 社区出边权重和（u→v且u∈社区,v∉社区）
        self.community_sum_in_total: Dict[str, float] = {}  # 社区总入边权重（所有指向社区的边）
        self.community_sum_in_out: Dict[str, float] = {}  # 社区总权重（sum_in + sum_out）
        self.node_in_degree: Dict[str, float] = {}  # 节点入度权重和
        self.node_out_degree: Dict[str, float] = {}  # 节点出度权重和
        self.node_community_links: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))  # 节点到社区的出边权重
        self._queue: Queue = Queue()  # 待处理节点队列
        self._processed: Set[str] = set()  # 已处理节点集合
        self._convergence_count: int = 0  # 收敛计数器
        self._max_convergence_rounds: int = 10  # 最大收敛轮次
        self.total_weight: float = 0.0  # 所有边权重和
        self.hierarchy_level = 0  # 层次迭代级别

        # 社区到节点的反向映射
        self.community_nodes: DefaultDict[str, Set[str]] = defaultdict(set)

        # 初始化社区映射跟踪
        self.final_community_map = {}  # 最终社区映射
        self.previous_level_communities = {}  # 上一层社区映射
        self.level_community_maps = []  # 每层的社区映射

        # 初始化源节点
        self._initialize_node(source)
        self.current_community[source] = source  # 确保源节点有初始社区
        self.final_community_map[source] = source  # 初始化最终映射
        self._queue.put(source)

    def _initialize_node(self, node: str):
        if node not in self.current_community:
            self.current_community[node] = node
            self.node_in_degree[node] = 0.0
            self.node_out_degree[node] = 0.0
            community = node
            self.community_sum_in[community] = 0.0
            self.community_sum_out[community] = 0.0
            self.community_sum_in_total[community] = 0.0
            self.community_sum_in_out[community] = 0.0
            self.community_nodes[community].add(node)

    def _calculate_modularity_gain(self, node: str, target_community: str) -> float:
        """计算模块化增益"""
        current_comm = self.current_community[node]
        if current_comm == target_community:
            return 0.0

        # 获取节点的出入度
        k_i_in = self.node_in_degree[node]
        k_i_out = self.node_out_degree[node]

        # 获取目标社区的总入度和总出度
        in_strength_tar = self.community_sum_in_total.get(target_community, 0.0)
        out_strength_tar = self.community_sum_out.get(target_community, 0.0)

        # 获取当前社区的总入度和总出度
        in_strength_curr = self.community_sum_in_total.get(current_comm, 0.0)
        out_strength_curr = self.community_sum_out.get(current_comm, 0.0)

        # 节点到目标社区的出边权重
        k_i_in_tar = self.node_community_links[node].get(target_community, 0.0)
        # 节点到当前社区的出边权重
        k_i_in_curr = self.node_community_links[node].get(current_comm, 0.0)

        # 总边权重
        m = self.total_weight if self.total_weight != 0 else 1e-9

        # 计算模块化增益
        delta = (k_i_in_tar / m) - (k_i_out * in_strength_tar / (m ** 2)) - \
                (k_i_in_curr / m) + (k_i_out * in_strength_curr / (m ** 2))

        return delta

    def _get_community_strength(self, community: str) -> float:
        """获取社区的总出度"""
        return self.community_sum_out.get(community, 0.0) + self.community_sum_in.get(community, 0.0)

    def _get_community_degree(self, community: str) -> float:
        """获取社区的总度数（入度+出度）"""
        return self.community_sum_in_out.get(community, 0.0)

    def push(self, node: str, edges: List[Dict[str, Any]], **kwargs):
        self._initialize_node(node)

        for e in edges:
            u, v = e['from'], e['to']
            weight = float(e.get('value', 0.0))
            self.total_weight += weight
            self._update_edge(u, v, weight)

        neighbor_edges = {e['to']: float(e['value']) for e in edges if e['from'] == node}  # 出边邻居
        self.node_out_degree[node] = sum(neighbor_edges.values())
        for e in edges:  # 处理入边以更新节点入度
            if e['to'] == node:
                self.node_in_degree[node] += float(e['value'])

        max_gain, best_community = self._find_best_community(neighbor_edges, node)
        if max_gain > 1e-9:
            self._move_node_to_community(node, best_community, neighbor_edges)
            self._convergence_count = 0
        else:
            self._processed.add(node)
            self._convergence_count += 1

    def _update_edge(self, u: str, v: str, weight: float):
        self._initialize_node(u)
        self._initialize_node(v)
        src_comm = self.current_community[u]
        dest_comm = self.current_community[v]

        # 更新节点出度
        self.node_out_degree[u] += weight

        # 更新节点到目标社区的出边权重
        prev_weight = self.node_community_links[u][dest_comm]
        self.node_community_links[u][dest_comm] += weight

        # 更新源社区的出边权重
        if src_comm == dest_comm:
            self.community_sum_in[src_comm] += weight  # 同社区内的边
        else:
            self.community_sum_out[src_comm] += weight  # 跨社区的出边
            # 更新目标社区的总入度
            self.community_sum_in_total[dest_comm] += weight

        # 更新社区总权重
        self.community_sum_in_out[src_comm] += weight
        self.community_sum_in_out[dest_comm] += weight

    def _find_best_community(self, neighbor_edges: Dict[str, float], node: str) -> Tuple[float, str]:
        """寻找最优目标社区"""
        max_gain = 0.0
        best_community = self.current_community[node]
        current_comm = self.current_community[node]

        # 考虑当前社区和所有邻居节点的社区
        candidate_communities = {current_comm}
        for neighbor in neighbor_edges:
            candidate_communities.add(self.current_community[neighbor])

        for community in candidate_communities:
            gain = self._calculate_modularity_gain(node, community)
            if gain > max_gain:
                max_gain = gain
                best_community = community

        return max_gain, best_community

    def _move_node_to_community(self, node: str, best_community: str, neighbor_edges: Dict[str, float]):
        """执行节点移动"""
        old_comm = self.current_community[node]
        if old_comm == best_community:
            return

        self._remove_node_from_community(node, old_comm)
        self._add_node_to_community(node, best_community, neighbor_edges)
        self._enqueue_neighbors(neighbor_edges.keys())

        # 更新最终社区映射
        self.final_community_map[node] = best_community

    def _remove_node_from_community(self, node: str, old_comm: str):
        """从原社区移除节点"""
        for comm, weight in list(self.node_community_links[node].items()):
            if comm == old_comm:
                self.community_sum_in[old_comm] -= weight
            else:
                self.community_sum_out[old_comm] -= weight

                self.community_sum_in_total[comm] -= weight

        for neighbor, weight in self.node_community_links[node].items():
            if neighbor == old_comm:
                continue

            self.node_community_links[neighbor][old_comm] -= weight

        # 更新社区总权重
        self.community_sum_in_out[old_comm] -= (self.node_out_degree[node] + self.node_in_degree[node])

        # 从社区节点集合中移除
        self.community_nodes[old_comm].discard(node)

    def _add_node_to_community(self, node: str, new_comm: str, neighbor_edges: Dict[str, float]):
        """将节点加入目标社区"""
        old_comm = self.current_community[node]
        self.current_community[node] = new_comm

        # 添加到社区节点集合
        self.community_nodes[new_comm].add(node)

        for neighbor, weight in neighbor_edges.items():
            neighbor_comm = self.current_community[neighbor]
            self.node_community_links[node][neighbor_comm] += weight
            if neighbor_comm == new_comm:
                self.community_sum_in[new_comm] += weight
            else:
                self.community_sum_out[new_comm] += weight
                self.community_sum_in_total[neighbor_comm] += weight

        for neighbor, weight in self.node_community_links[node].items():
            if neighbor == new_comm:
                continue
            self.node_community_links[neighbor][new_comm] += weight

        # 更新社区总权重
        self.community_sum_in_out[new_comm] += (self.node_out_degree[node] + self.node_in_degree[node])

    def _enqueue_neighbors(self, neighbors: Set[str]):
        """将邻居节点加入队列"""
        for neighbor in neighbors:
            if neighbor not in self._processed:
                self._queue.put(neighbor)

    def pop(self) -> Dict[str, Any]:
        """获取待处理节点"""
        if self._convergence_count >= self._max_convergence_rounds:
            return None

        while not self._queue.empty():
            node = self._queue.get()
            if node not in self._processed:
                return {'node': node}
        return None

    def aggregate_communities(self):
        """构建有向元网络并进行下一轮迭代"""
        # 保存当前层社区映射
        self.previous_level_communities = self.current_community.copy()
        self.level_community_maps.append(self.current_community.copy())

        # 构建社区映射
        community_map = defaultdict(list)
        for node, comm in self.current_community.items():
            community_map[comm].append(node)

        # 构建元网络
        meta_graph = nx.MultiDiGraph()

        # 添加元节点（社区）
        for comm in community_map:
            meta_graph.add_node(comm)

        # 添加元边（社区间的连接）
        for node in self.current_community:
            src_comm = self.current_community[node]
            for tgt_comm, weight in self.node_community_links[node].items():
                if src_comm != tgt_comm and weight > 0:
                    meta_graph.add_edge(src_comm, tgt_comm, value=weight)

        # 保存当前层到上一层的映射
        level_mapping = {}
        for meta_node, original_nodes in community_map.items():
            level_mapping[meta_node] = original_nodes

        # 重置参数，准备下一轮迭代
        self._reset_for_next_level()

        # 返回元网络、社区映射和层间映射
        return meta_graph, community_map, level_mapping

    def _reset_for_next_level(self):
        """重置参数以进行下一轮迭代"""
        # 保留当前社区划分作为上一层的结果
        self.hierarchy_level += 1

        # 重置当前社区参数
        self.current_community = {}
        self.community_sum_in = {}
        self.community_sum_out = {}
        self.community_sum_in_total = {}
        self.community_sum_in_out = {}
        self.node_in_degree = {}
        self.node_out_degree = {}
        self.node_community_links = defaultdict(lambda: defaultdict(float))
        self._queue = Queue()
        self._processed = set()
        self._convergence_count = 0
        self.total_weight = 0.0
        self.community_nodes = defaultdict(set)


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

        model = self.model_cls(src)

        edges = list()
        for edge_view in [
            graph.in_edges(src, data=True),
            graph.out_edges(src, data=True)
        ]:
            for u, v, attrs in edge_view:
                edges.append({'from': u, 'to': v, **attrs})
        model.push(src, edges)

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

            pop_item = model.pop()
            if pop_item is not None:
                node = pop_item['node'] if isinstance(pop_item, dict) else None
                vis.add(node)

        # 执行社区聚合（多层迭代）
        max_levels = 5  # 限制最大迭代层数
        current_level = 0
        level_mappings = []  # 保存每层的映射关系

        # 保存最终的社区映射到模型属性中
        final_community_map = model.current_community

        # 记录每层的社区数量
        prev_community_count = len(set(model.current_community.values()))

        while current_level < max_levels:
            meta_graph, community_map, level_mapping = model.aggregate_communities()
            level_mappings.append(level_mapping)

            # 检查社区数量是否变化
            current_community_count = len(meta_graph.nodes())
            if current_community_count == prev_community_count:
                break  # 社区数量不再变化，收敛
            prev_community_count = current_community_count

            # 在元网络上重新初始化模型
            new_model = self.model_cls(list(meta_graph.nodes())[0])
            for u, v, data in meta_graph.edges(data=True):
                edges = [{'from': u, 'to': v, 'value': data.get('value', 1.0)}]
                new_model.push(u, edges)

            model = new_model
            current_level += 1

        # 更新最终社区映射（将元社区映射回原始节点）
        if current_level > 0:
            final_community_map = {}
            last_level_map = model.current_community

            for meta_node, original_nodes in level_mappings[-1].items():
                meta_community = last_level_map.get(meta_node, meta_node)
                for node in original_nodes:
                    final_community_map[node] = meta_community

            for level_map in reversed(level_mappings[:-1]):
                temp_map = {}
                for meta_node, original_nodes in level_map.items():
                    meta_community = final_community_map.get(meta_node, meta_node)
                    for node in original_nodes:
                        temp_map[node] = meta_community
                final_community_map = temp_map

        model.final_community_map = final_community_map
        model.level_community_maps = level_mappings

        return graph.subgraph(list(vis))
