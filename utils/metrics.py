from typing import Dict, Set

import networkx as nx


def calc_recall(witness_graph: nx.MultiDiGraph, targets) -> float:
    assert len(targets) > 0

    target_cnt = 0
    for target in targets:
        if witness_graph.has_node(target):
            target_cnt += 1

    return target_cnt / len(targets)


def calc_coverage(
        witness_graph: nx.MultiDiGraph,
        graph: nx.MultiDiGraph,
        addr2label: Dict[str, str],
) -> float:
    # TODO
    pass


def calc_size(witness_graph: nx.MultiDiGraph) -> int:
    return witness_graph.number_of_nodes()


def calc_depth(vis_labels: Set) -> int:
    vis_labels = [
        label for label in vis_labels
        if label.startswith('ml_transit_')
    ]
    node_depths = [
        int(label.replace('ml_transit_', ''))
        for label in vis_labels
    ]
    return max(node_depths)
