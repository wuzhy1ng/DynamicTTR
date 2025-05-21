import argparse
import csv
import os.path
import re
import time
from datetime import datetime
from typing import Tuple, Any, Callable

import networkx as nx
from tqdm import tqdm

from algos.dttr import DTTR
from dataset.dynamic import DynamicTransNetwork
from settings import PROJECT_PATH
from utils.metrics import calc_depth, calc_recall, calc_size, calc_precision


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
    model = model_cls(source=list(sources), **kwargs)
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
    return (depth, recall, precision, num_nodes, time_used)


def eval_method(
        dataset: DynamicTransNetwork,
        model_cls: Any,
        eval_fn: Callable,
        **kwargs
) -> Tuple:
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
        depth, recall, precision, num_nodes, time_used = eval_fn(
            dataset=dataset,
            case_name=name,
            model_cls=model_cls,
            **kwargs,
        )
        print(
            datetime.now(),
            '{}: {}[depth], {}[recall], {}[precision] {}[#nodes], {}[#seconds]'.format(
                name, depth, recall, precision, num_nodes, time_used
            )
        )
        count = dataset.get_case_transaction_count(name)
        list_depth.append(depth + 1)
        list_recalls.append(recall)
        list_precisions.append(precision)
        list_num_nodes.append(num_nodes)
        list_time_used.append(time_used)
        list_tps.append(count / (time_used if time_used > 0 else 1))

    avg_depth = sum(list_depth) / len(list_depth)
    avg_recall = sum(list_recalls) / len(list_recalls)
    avg_precision = sum(list_precisions) / len(list_precisions)
    size = sum(list_num_nodes) / len(list_num_nodes)
    tps = sum(list_tps) / len(list_tps)
    return avg_depth, avg_recall, avg_precision, size, tps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    args = parser.parse_args()

    dataset = DynamicTransNetwork(raw_path=args.raw_path)
    cached_result_path = os.path.join(PROJECT_PATH, 'cache', 'sens_exps.csv')
    file = open(cached_result_path, 'w', encoding='utf-8', newline='\n')
    writer = csv.writer(file)
    writer.writerow(['alpha', 'epsilon', 'depth', 'recall', 'precision', 'size', 'tps'])
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25]:
        for epsilon in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
            print('alpha:', alpha, 'epsilon:', epsilon)
            avg_depth, avg_recall, avg_precision, size, tps = eval_method(
                dataset=dataset,
                model_cls=DTTR,
                eval_fn=eval_case_from_transaction_arrive,
                alpha=alpha,
                epsilon=epsilon,
            )
            print(
                alpha, epsilon,
                avg_depth, avg_recall, avg_precision, size, tps
            )
            writer.writerow([
                alpha, epsilon,
                avg_depth, avg_recall, avg_precision, size, tps
            ])
    file.close()
