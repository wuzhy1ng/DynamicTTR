import argparse
import time
from datetime import datetime
from typing import Tuple, Any, Callable

import networkx as nx
from tqdm import tqdm

from algos.bfs import BFS
from algos.appr import APPR
from algos.dappr import DAPPR
from algos.dttr import DTTR
from algos.haricut import Haircut
from algos.louvain import LOUVAIN
from algos.lpa import LPA
from algos.poison import Poison
from algos.push_pop import PushPopAggregator
from algos.tiles import TILES
from algos.tpp import TPP
from algos.ttr import TTRRedirect

from utils.metrics import calc_depth, calc_recall, calc_size, calc_precision, calc_fpr
from dataset.dynamic import DynamicTransNetwork
import re


def eval_case_from_pushpop(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any,
        **kwargs
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
    transits, targets = set(), set()
    pattern = r"ml_transit_.*?"
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)
        if re.match(pattern, str(label)):
            transits.add(addr)
            continue
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
    source = sorted(list(sources))[-1]
    aggregator = PushPopAggregator(
        source=source,
        model_cls=model_cls,
    )
    witness_graph = aggregator.execute(graph)
    time_used += (time.time() - s_time)
    vis = set(list(witness_graph.nodes()))

    # collect the metrics and return the result
    depth = calc_depth(witness_graph, [source], vis)
    recall = calc_recall(witness_graph, transits)
    precision = calc_precision(witness_graph, transits, targets)
    fpr = calc_fpr(witness_graph, transits, targets)
    num_nodes = calc_size(witness_graph)
    tps = 1 / (time_used if time_used > 0 else 1e-5)
    return (depth, recall, precision, fpr, num_nodes, tps)


def eval_case_from_edge_arrive(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any = DTTR,
        **kwargs
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
    transits, targets = set(), set()
    pattern = r"ml_transit_.*?"
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)
        if re.match(pattern, str(label)):
            transits.add(addr)
            continue
        targets.add(addr)

    # build the snapshot network from arrived edges
    time_used = time.time()
    model = model_cls(source=list(sources), **kwargs)
    graph = nx.MultiDiGraph()
    for u, v, attr in tqdm(
            iterable=dataset.iter_edge_arrive(case_name),
            desc=case_name,
    ):
        graph.add_edge(u, v, **attr)
        model.edge_arrive(u, v, attr)
    vis = list(model.p.keys())
    witness_graph = graph.subgraph(vis)
    time_used = time.time() - time_used

    # collect the metrics and return the result
    depth = calc_depth(witness_graph, sources, vis)
    recall = calc_recall(witness_graph, transits)
    precision = calc_precision(witness_graph, transits, targets)
    fpr = calc_fpr(witness_graph, transits, targets)
    num_nodes = calc_size(witness_graph)
    tps = dataset.get_case_transaction_count(case_name) / (time_used if time_used > 0 else 1e-5)
    return (depth, recall, precision, fpr, num_nodes, tps)


def eval_case_from_transaction_arrive(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any = DTTR,
        **kwargs,
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
    transits, targets = set(), set()
    pattern = r"ml_transit_.*?"
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)
        if re.match(pattern, str(label)):
            transits.add(addr)
            continue
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
    time_used = time.time()
    model = model_cls(source=list(sources), **kwargs)
    graph = nx.MultiDiGraph()
    for txhash in tqdm(
            iterable=txhash_sorted,
            desc=case_name,
    ):
        graph.add_edges_from(trans2edges[txhash])
        trans = nx.MultiDiGraph()
        trans.add_edges_from(trans2edges[txhash])
        model.transaction_arrive(trans)
    time_used = time.time() - time_used
    vis = list(model.p.keys())
    witness_graph = graph.subgraph(vis)

    # collect the metrics and return the result
    depth = calc_depth(witness_graph, sources, vis)
    recall = calc_recall(witness_graph, transits)
    precision = calc_precision(witness_graph, transits, targets)
    fpr = calc_fpr(witness_graph, transits, targets)
    num_nodes = calc_size(witness_graph)
    tps = dataset.get_case_transaction_count(case_name) / (time_used if time_used > 0 else 1e-5)
    return (depth, recall, precision, fpr, num_nodes, tps)


def eval_method(
        dataset: DynamicTransNetwork,
        model_cls: Any,
        eval_fn: Callable,
        **kwargs
):
    """
    Perform the whole evaluation for the given method
    using the function `eval_fn`.

    :param dataset: the dynamic network
    :param model_cls: the method class to be evaluated
    :param eval_fn: the case evaluation function
    :return:
    """
    (list_depth, list_recalls, list_precisions, list_fprs,
     list_num_nodes, list_time_used, list_tps) = [
        list() for _ in range(7)
    ]
    for name in dataset.get_case_names():
        depth, recall, precision, fpr, num_nodes, tps = eval_fn(
            dataset=dataset,
            case_name=name,
            model_cls=model_cls,
            **kwargs,
        )
        print(
            datetime.now(),
            '{}: {}[depth], {}[recall], {}[precision], {}[fpr], {}[#nodes], {}[tps]'.format(
                name, depth, recall, precision, fpr, num_nodes, tps
            )
        )
        list_depth.append(depth)
        list_recalls.append(recall)
        list_precisions.append(precision)
        list_fprs.append(fpr)
        list_num_nodes.append(num_nodes)
        list_tps.append(tps)

    avg_depth = sum(list_depth) / len(list_depth)
    avg_recall = sum(list_recalls) / len(list_recalls)
    avg_precision = sum(list_precisions) / len(list_precisions)
    avg_fpr = sum(list_fprs) / len(list_fprs)
    size = sum(list_num_nodes) / len(list_num_nodes)
    tps = min(list_tps)

    print('======= Evaluation: {} ======='.format(model_cls))
    print('Depth: {} (mean)'.format(avg_depth), list_depth)
    print('Recall: {} (mean)'.format(avg_recall), list_recalls)
    print('Precision: {} (mean)'.format(avg_precision), list_precisions)
    print('FPR: {} (mean)'.format(avg_fpr), list_fprs)
    print('Num Nodes: {} (mean)'.format(size), list_num_nodes)
    print('TPS: {} (min)'.format(tps), list_tps)

    return avg_depth, avg_recall, avg_precision, avg_fpr, size, tps


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
        model_cls=TILES,
        eval_fn=eval_case_from_edge_arrive,
    )
    for model_cls in [
        BFS, Poison,
        Haircut,
        APPR,
        TTRRedirect,
        LPA, LOUVAIN, TPP,
    ]:
        eval_method(
            dataset=dataset,
            model_cls=model_cls,
            eval_fn=eval_case_from_pushpop
        )
