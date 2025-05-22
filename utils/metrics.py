from typing import Set

import networkx as nx


def calc_precision(
        witness_graph: nx.MultiDiGraph,
        transits: Set, targets: Set
) -> float:
    nodes = set(list(witness_graph.nodes()))
    tp = len(transits.intersection(nodes - targets))
    fp = len(nodes - transits - targets)
    if tp == 0:
        return 0
    return tp / (tp + fp)


def calc_recall(
        witness_graph: nx.MultiDiGraph,
        transits: Set,
) -> float:
    assert len(transits) > 0
    hits = set(list(witness_graph.nodes())).intersection(transits)
    return len(hits) / len(transits)


def calc_fpr(
        witness_graph: nx.MultiDiGraph,
        transits: Set, targets: Set
) -> float:
    nodes = set(list(witness_graph.nodes()))
    fp = len(nodes - transits - targets)
    tn = len(nodes.intersection(targets))
    if fp == 0:
        return 0
    return fp / (fp + tn)


def calc_size(witness_graph: nx.MultiDiGraph) -> int:
    return witness_graph.number_of_nodes()


def calc_depth(graph: nx.MultiDiGraph, sources, vis) -> int:
    max_depth = 0
    for source in sources:
        graph = graph.to_undirected()
        paths = nx.single_source_shortest_path(graph, source)
        path_lens = [
            len(path) for target, path in paths.items()
            if target in vis
        ]
        depth = max(path_lens)
        if depth > max_depth:
            max_depth = depth
    return max_depth
