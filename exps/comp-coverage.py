import argparse
import os
import sys
import time
from datetime import datetime
from typing import Tuple, Any, Callable

import networkx as nx
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('/', 1)[0])

from algos.bfs import BFS
from algos.appr import APPR
from algos.dttr import DTTR
from algos.haricut import Haircut
from algos.poison import Poison
from algos.push_pop import PushPopAggregator
from algos.ttr import TTRRedirect
import decimal

from utils.metrics import calc_depth, calc_recall, calc_size, calc_coverage
from dataset.dynamic import DynamicTransNetwork
import re


def eval_case_from_pushpop(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any,
) -> Tuple:
    """
    Perform the evaluation with the `PushPopModel`.

    :param dataset: the dynamic network
    :param case_name: the tracing case name
    :param model_cls: the class of the PushPopModel method
    :return: the collected evaluating metrics
    """

    # init the source
    sources = set()
    targets = set()
    pattern = r"ml_transit_.*?"
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)
        if re.match(pattern, str(label)):
            targets.add(addr)

    # build the snapshot network from arrived edges
    graph = nx.MultiDiGraph()
    time_used = 0
    for u, v, attr in tqdm(
            iterable=dataset.iter_edge_arrive(case_name),
            desc=case_name,
    ):
        attr['value'] = decimal.Decimal(attr['value'])
        graph.add_edge(u, v, **attr)
        # if u not in vis and v not in vis:
        #     continue
    s_time = time.time()
    aggregator = PushPopAggregator(
        source_list=list(sources),
        model_cls=model_cls,
    )
    witness_graph = aggregator.execute(graph)
    vis = set([n for n in witness_graph.nodes()])
    time_used += (time.time() - s_time)
    witness_graph = graph.subgraph(list(vis))

    # collect the metrics and return the result
    coverage = calc_coverage(witness_graph, graph, sources)
    print("coverage= ", coverage)
    depth = calc_depth({addr2label.get(addr) for addr in vis})
    recall = calc_recall(witness_graph, list(targets))
    num_nodes = calc_size(witness_graph)
    return (coverage, depth, recall, num_nodes, time_used)


def eval_case_from_edge_arrive(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any = DTTR,
) -> Tuple:
    """
    Perform the evaluation with the edge arrived method.

    :param dataset: the dynamic network
    :param case_name: the tracing case name
    :param model_cls: the class of the edge arrived method
    :return: the collected evaluating metrics
    """
    # init the source
    sources = set()
    targets = set()
    pattern = r"ml_transit_.*?"
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)
        if re.match(pattern, str(label)):
            targets.add(addr)  ######## sir this way

    # build the snapshot network from arrived edges
    time_used = 0
    model = model_cls(source=list(sources))
    graph = nx.MultiDiGraph()
    for u, v, attr in tqdm(
            iterable=dataset.iter_edge_arrive(case_name),
            desc=case_name,
    ):
        graph.add_edge(u, v, **attr)
        s_time = time.time()
        model.edge_arrive(u, v, attr)
        time_used += (time.time() - s_time)
    vis = list(model.p.keys())
    witness_graph = graph.subgraph(vis)

    # collect the metrics and return the result
    depth = calc_depth({addr2label.get(addr) for addr in vis})
    recall = calc_recall(witness_graph, list(targets))
    num_nodes = calc_size(witness_graph)
    return (depth, recall, num_nodes, time_used)


def eval_method(
        dataset: DynamicTransNetwork,
        model_cls: Any,
        eval_fn: Callable,
):
    """
    Perform the whole evaluation for the given method
    using the function `eval_fn`.

    :param dataset: the dynamic network
    :param model_cls: the method class to be evaluated
    :param eval_fn: the case evaluation function
    :return:
    """
    print('======= Evaluating: {} ======='.format(model_cls))
    list_coverage, list_depth, list_recalls, list_num_nodes, list_time_used = [
        list() for _ in range(5)
    ]
    for name in dataset.get_case_names():
        print(datetime.now(), 'processing:', name)
        coverage, depth, recall, num_nodes, time_used = eval_fn(
            dataset=dataset,
            case_name=name,
            model_cls=model_cls,
        )
        list_coverage.append(coverage)
        list_depth.append(depth)
        list_recalls.append(recall)
        list_num_nodes.append(num_nodes)
        list_time_used.append(time_used)

    print('Coverage: {} (mean)'.format(sum(list_coverage) / len(list_coverage)), list_coverage)
    print('Depth: {} (mean)'.format(sum(list_depth) / len(list_depth)), list_depth)
    print('Recall: {} (mean)'.format(sum(list_recalls) / len(list_recalls)), list_recalls)
    print('Num Nodes: {} (mean)'.format(sum(list_num_nodes) / len(list_num_nodes)), list_num_nodes)
    print('Used Time: {} (mean)'.format(sum(list_time_used) / len(list_time_used)), list_time_used)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    # args = parser.parse_args()

    # dataset = DynamicTransNetwork(raw_path=args.raw_path)
    dataset = DynamicTransNetwork(raw_path="./data2")
    # eval_method(
    #     dataset=dataset,
    #     model_cls=DTTR,
    #     eval_fn=eval_case_from_edge_arrive,
    # )
    for model_cls in [
       TTRRedirect
    ]:
        eval_method(
            dataset=dataset,
            model_cls=model_cls,
            eval_fn=eval_case_from_pushpop
        )
