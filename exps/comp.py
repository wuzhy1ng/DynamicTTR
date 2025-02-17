import argparse
import time
from typing import Tuple

import networkx as nx

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('/', 1)[0])

from utils.metrics import calc_depth, calc_recall, calc_size
from algos.bfs import BFS
from algos.push_pop import PushPopModel
from dataset.dynamic import DynamicTransNetwork
import re

def eval_case(
        dataset: DynamicTransNetwork,
        case_name: str,
        method: PushPopModel
) -> Tuple:
    # init the source
    vis = set()
    targets = set()
    pattern = r"ml_transit_[0-9]"
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            vis.add(addr)
        if re.match(pattern, str(label)):
            targets.add(addr) ######## sir this way
    method = BFS(vis)

    # build the snapshot network from arrived edges
    graph = nx.MultiDiGraph()
    time_used = 0
    for u, v, attr in dataset.iter_edge_arrive(case_name):
        graph.add_edge(u, v, **attr)
        if u not in vis and v not in vis:
            continue
        s_time = time.time()
        witness_graph = method.execute(graph)
        vis = set([n for n in witness_graph.nodes()])
        time_used += (time.time() - s_time)
    witness_graph = graph.subgraph(list(vis))

    depth = calc_depth(witness_graph, vis)
    recall = calc_recall(witness_graph, list(targets))
    num_nodes = calc_size(witness_graph)
    return (depth, recall, num_nodes, time_used)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    args = parser.parse_args()

    dataset = DynamicTransNetwork(raw_path=args.raw_path)
    list_depth, list_recalls, list_num_nodes, list_time_used = [
        list() for _ in range(4)
    ]
    for name in dataset.get_case_names():
        depth, recall, num_nodes, time_used = eval_case(
            dataset=dataset,
            case_name=name,
            method=BFS(None)
        )
        list_depth.append(depth)
        list_recalls.append(recall)
        list_num_nodes.append(num_nodes)
        list_time_used.append(time_used)

    print('Method:', 'BFS')
    print('Depth:', list_depth)
    print('Recall:', list_recalls)
    print('Num Nodes:', list_num_nodes)
    print('Time:', list_time_used)
