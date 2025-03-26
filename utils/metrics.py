from typing import Dict, Set
import decimal

import networkx as nx
from utils.price import get_usd_value


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
        for u, v, _, attr in graph.out_edges(src, keys=True, data=True):
            all_dirty_value += decimal.Decimal(get_usd_value(attr["contractAddress"], attr['value'], attr['timeStamp']))
            print("addr:", attr['contractAddress'])
            print("value:", attr['value'])
            print("timeStamp:", attr['timeStamp'])
            print("all dirty value: ", all_dirty_value)

    # witness_graph的所有流出金额
    all_out_value = 0
    # for u, v, _, attr in graph.out_edges(keys=True, data=True):
    #     if u in witness_graph and not v in witness_graph:
    #         all_out_value += decimal.Decimal(get_usd_value(attr["contractAddress"], attr['value'], attr['timeStamp']))
    # print("all out value:", all_out_value)
    
    # witness_graph中的余额
    bal = dict()
    for u, v, _, attr in witness_graph.in_edges(keys=True, data=True):
        usd_val = decimal.Decimal(get_usd_value(attr["contractAddress"], attr['value'], attr['timeStamp']))
    
        bal[v] = bal.get(v, decimal.Decimal(0)) + usd_val
        bal[u] = bal.get(u, decimal.Decimal(0)) - usd_val

    for k in bal:
        if bal[k]<0:
            bal[k] = 0
    all_bal = sum(bal.values())
    print("all balance", all_bal)

    if all_dirty_value == 0:
        return 0
    actual_coverage = (all_bal+all_out_value)/all_dirty_value
    print("actual coverage", actual_coverage)

    # return 1 if actual_coverage>1 else actual_coverage
    return actual_coverage

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
