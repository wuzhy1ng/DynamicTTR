import argparse
import time
from datetime import datetime
from typing import Tuple, Any, Callable

import networkx as nx
from tqdm import tqdm

from algos.bfs import BFS
from algos.appr import APPR
from algos.dttr import DTTR
from algos.haricut import Haircut
from algos.poison import Poison
from algos.push_pop import PushPopAggregator
from algos.ttr import TTRRedirect

from utils.metrics import calc_depth, calc_recall, calc_size, calc_precision
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
    s_time = time.time()
    graph = nx.MultiDiGraph()
    time_used = 0
    for u, v, attr in tqdm(
            iterable=dataset.iter_edge_arrive(case_name),
            desc=case_name,
    ):
        attr['value'] = float(attr['value'])
        graph.add_edge(u, v, **attr)
        # if u not in vis and v not in vis:
        #     continue
    source = sorted(list(sources))[0]
    aggregator = PushPopAggregator(
        source=source,
        model_cls=model_cls,
    )
    witness_graph = aggregator.execute(graph)
    time_used += (time.time() - s_time)

    # collect the metrics and return the result
    depth = calc_depth(witness_graph, [source])
    recall = calc_recall(witness_graph, targets)
    precision = calc_precision(witness_graph, targets)
    num_nodes = calc_size(witness_graph)
    tps = 1 / (time_used if time_used > 0 else 1)
    return (depth, recall, precision, num_nodes, tps)


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
            targets.add(addr)

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
    depth = calc_depth(witness_graph, sources)
    recall = calc_recall(witness_graph, targets)
    precision = calc_precision(witness_graph, targets)
    num_nodes = calc_size(witness_graph)
    tps = dataset.get_case_transaction_count(case_name) / (time_used if time_used > 0 else 1)
    return (depth, recall, precision, num_nodes, tps)


def eval_case_from_transaction_arrive(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any = DTTR,
) -> Tuple:
    """
    Perform the evaluation with the transaction arrived method.

    :param dataset: the dynamic network
    :param case_name: the tracing case name
    :param model_cls: the class of the transaction arrived method
    :return: the collected evaluating metrics
    """
    # init the source
    sources = set()
    targets = set()
    pattern = r"ml_transit_(\d+)"
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)
        if re.match(pattern, str(label)):
            targets.add(addr)

    # build the time-ordered trans. from arrived edges
    trans2time, trans2edges = dict(), dict()
    for u, v, attr in dataset.iter_edge_arrive(case_name):
        txhash = attr['hash']
        if trans2edges.get(txhash) is None:
            trans2edges[txhash] = list()
        trans2edges[txhash].append((u, v, attr))
        if trans2time.get(txhash):
            continue
        trans2time[txhash] = attr['timeStamp']
    txhash_sorted = sorted(list(trans2time.items()), key=lambda x: x[1])
    txhash_sorted = [txhash for txhash, _ in txhash_sorted]

    # perform trans. arrive operations
    time_used = 0
    model = model_cls(source=list(sources))
    graph = nx.MultiDiGraph()
    for txhash in tqdm(
            iterable=txhash_sorted,
            desc=case_name,
    ):
        graph.add_edges_from(trans2edges[txhash])
        trans = nx.MultiDiGraph()
        trans.add_edges_from(trans2edges[txhash])
        s_time = time.time()
        model.transaction_arrive(trans)
        time_used += (time.time() - s_time)
    vis = list(model.p.keys())
    # vis = [
    #     node for node in model.p.keys()
    #     if model.p[node] >= model.epsilon
    # ]
    witness_graph = graph.subgraph(vis)

    # collect the metrics and return the result
    depth = calc_depth(witness_graph, sources)
    recall = calc_recall(witness_graph, targets)
    precision = calc_precision(witness_graph, targets)
    num_nodes = calc_size(witness_graph)
    tps = dataset.get_case_transaction_count(case_name) / (time_used if time_used > 0 else 1)
    return (depth, recall, precision, num_nodes, tps)


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
    (list_depth, list_recalls, list_precisions,
     list_num_nodes, list_time_used, list_tps) = [
        list() for _ in range(6)
    ]
    for name in dataset.get_case_names():
        depth, recall, precision, num_nodes, tps = eval_fn(
            dataset=dataset,
            case_name=name,
            model_cls=model_cls,
        )
        print(
            datetime.now(),
            '{}: {}[depth], {}[recall], {}[precision] {}[#nodes], {}[tps]'.format(
                name, depth, recall, precision, num_nodes, tps
            )
        )
        list_depth.append(depth + 1)
        list_recalls.append(recall)
        list_precisions.append(precision)
        list_num_nodes.append(num_nodes)
        list_tps.append(tps)

    print('======= Evaluation: {} ======='.format(model_cls))
    print('Depth: {} (mean)'.format(sum(list_depth) / len(list_depth)), list_depth)
    print('Recall: {} (mean)'.format(sum(list_recalls) / len(list_recalls)), list_recalls)
    print('Precision: {} (mean)'.format(sum(list_precisions) / len(list_precisions)), list_precisions)
    print('Num Nodes: {} (mean)'.format(sum(list_num_nodes) / len(list_num_nodes)), list_num_nodes)
    print('TPS: {} (mean)'.format(min(list_tps)), list_tps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    args = parser.parse_args()

    dataset = DynamicTransNetwork(raw_path=args.raw_path)
    eval_method(
        dataset=dataset,
        model_cls=DTTR,
        eval_fn=eval_case_from_transaction_arrive,
    )
    eval_method(
        dataset=dataset,
        model_cls=DTTR,
        eval_fn=eval_case_from_edge_arrive,
    )
    for model_cls in [
        BFS, Poison,
        Haircut,
        APPR,
        TTRRedirect
    ]:
        eval_method(
            dataset=dataset,
            model_cls=model_cls,
            eval_fn=eval_case_from_pushpop
        )
