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
        sources: Set[str],
) -> float:
    # ml_transit_0的出边金额总和
    all_dirty_value = 0
    for src in sources:
        for _, _, attr in graph.out_edges(src, keys=True, data=True):
            all_dirty_value += float(attr['value'])

    # witness_graph的所有流出金额
    all_out_value = 0
    for u, v, attr in graph.out_edges(data=True):
        if u in witness_graph and not v in witness_graph:
            all_out_value += float(attr['value'])
    
    # witness_graph中的余额
    bal = dict()
    for u, v, attr in graph.in_edges(data=True):
        if v in witness_graph:
            if bal.get(v):
                bal[v] += float(attr['value'])
            else:
                bal[v] = float(attr['value'])
    for u, v, attr in graph.out_edges(data=True):
        if u in witness_graph:
            if bal.get(u):
                bal[u] = 0 if bal[u]<float(attr['value']) else bal[u]-float(attr['value'])
            else:
                bal[u] = 0
    
    return (sum(bal.values())+all_out_value)/all_dirty_value

def calc_size(witness_graph: nx.MultiDiGraph) -> int:
    return witness_graph.number_of_nodes()


def calc_depth(vis_labels: Set) -> int:
    vis_labels = [
        label for label in vis_labels
        if label and label.startswith('ml_transit_')
    ]
    node_depths = [
        int(label.replace('ml_transit_', ''))
        for label in vis_labels
    ]
    return 0 if node_depths == [] else max(node_depths)
