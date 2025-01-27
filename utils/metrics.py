import networkx as nx
from typing import Tuple

class GraphMetrics:
    def __init__(self, dataset, case_name):
        self.dataset = dataset
        self.case_name = case_name

    def calculate_depth(self, witness_graph: nx.MultiDiGraph) -> int:

        if not witness_graph.nodes():
            return 0

        max_depth = 0
        for node in witness_graph.nodes():
            try:
                lengths = nx.single_source_shortest_path_length(witness_graph, node)
                current_max = max(lengths.values())
                if current_max > max_depth:
                    max_depth = current_max
            except nx.NetworkXNoPath:
                continue

        return max_depth

    def calculate_recall(self, witness_graph: nx.MultiDiGraph) -> float:

        addr2label = self.dataset.get_case_labels(self.case_name)
        all_positive_nodes = {addr for addr, label in addr2label.items() if label == 'ml_transit_0'}
        witness_nodes = set(witness_graph.nodes())
        positive_nodes_in_witness = all_positive_nodes.intersection(witness_nodes)

        if not all_positive_nodes:
            return 0.0

        recall = len(positive_nodes_in_witness) / len(all_positive_nodes)
        return recall

    def calculate_num_nodes(self, witness_graph: nx.MultiDiGraph) -> int:

        return witness_graph.number_of_nodes()

    def calc_metrics(self, witness_graph: nx.MultiDiGraph) -> Tuple[int, float, int]:

        depth = self.calculate_depth(witness_graph)
        recall = self.calculate_recall(witness_graph)
        num_nodes = self.calculate_num_nodes(witness_graph)

        return depth, recall, num_nodes
