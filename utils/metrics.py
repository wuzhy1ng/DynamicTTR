import networkx as nx
from requests.packages import target


def calc_recall(g: nx.Graph, targets) -> float:
    assert len(targets) > 0

    target_cnt = 0
    for target in targets:
        if g.has_node(target):
            target_cnt += 1

    return target_cnt / len(targets)


def calc_size(g: nx.Graph) -> int:
    return g.number_of_nodes()


def calc_depth(sources: set, targets: set) -> int:
    nodes = targets.intersection(sources)
    node_depths = [
        int(node.replace('ml_transit_', ''))
        for node in nodes
    ]
    return max(node_depths)
