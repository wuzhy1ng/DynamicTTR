import networkx as nx

def calc_recall(g: nx.Graph, targets: list) -> float:
    assert len(targets) > 0

    target_cnt = 0
    for target in targets:
        if g.has_node(target):
            target_cnt += 1

    return target_cnt / len(targets)


def calc_size(g: nx.Graph) -> int:
    return g.number_of_nodes()


def calc_depth(g: nx.Graph, sources) -> int:
    max_depth = 0
    for source in sources:
        # seems that there's a multi_source_function
        K = nx.single_source_shortest_path_length(g, source)
        K = list(K.items())
        K.sort(key=lambda x: x[1], reverse=True)
        max_depth = max(max_depth, K[0][1] if len(K) > 0 else 0)
    return max_depth