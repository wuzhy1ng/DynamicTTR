import argparse
import csv
import os.path
import time
from typing import Any, List, Dict

import networkx as nx
from tqdm import tqdm

from algos.appr import APPR
from algos.bfs import BFS
from algos.dappr import DAPPR
from algos.dttr import DTTR
from algos.haricut import Haircut
from algos.poison import Poison
from algos.push_pop import PushPopAggregator
from algos.tiles import TILES
from algos.ttr import TTRRedirect
from dataset.dynamic import DynamicTransNetwork
from settings import PROJECT_PATH


def eval_tps_from_pushpop(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any,
        record_points: List[float],
        transaction_cnt: int,
        **kwargs
) -> List:
    """
    Perform the evaluation with the `PushPopModel`.

    :param dataset: the dynamic network
    :param case_name: the tracing case name
    :param model_cls: the class of the PushPopModel method
    :return: the collected evaluating metrics
    """
    results = list()

    # init the source
    sources = set()
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)

    # build the snapshot network from arrived edges
    # the loading time should be recorded,
    # because the pushpop models are offline algorithms
    point_idx = 0
    graph = nx.MultiDiGraph()
    loading_time = time.time()
    data = [(u, v, attr) for u, v, attr in dataset.iter_edge_arrive(case_name)]
    loading_time = time.time() - loading_time

    # run the model
    processor = tqdm(iterable=data, desc=str(model_cls), total=len(data))
    for u, v, attr in processor:
        attr['value'] = float(attr['value'])
        graph.add_edge(u, v, **attr)
        source = sorted(list(sources))[0]
        aggregator = PushPopAggregator(
            source=source,
            model_cls=model_cls,
        )
        aggregator.execute(graph)

        process = (processor.n + 1) / processor.total
        current_speed = processor.format_dict["rate"]
        if current_speed is None:
            continue
        current_speed = 1 / current_speed
        current_speed += loading_time * process
        current_speed = 1 / current_speed
        current_speed *= (len(data) / transaction_cnt)
        if record_points[point_idx] <= process:
            results.append(current_speed)
            print(model_cls, record_points[point_idx], current_speed)
            point_idx += 1
        if current_speed < 1:
            return results + [1 for _ in range(len(record_points) - point_idx)]
        attr['value'] = float(attr['value'])
    return results


def eval_tps_from_edge_arrive(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any,
        record_points: List[float],
        transaction_cnt: int,
        **kwargs
) -> List:
    """
    Perform the evaluation with the edge arrived method.

    :param dataset: the dynamic network
    :param case_name: the tracing case name
    :param model_cls: the class of the edge arrived method
    :return: the collected evaluating metrics
    """
    results = list()

    # init the source
    sources = set()
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)

    # build the snapshot network from arrived edges
    point_idx = 0
    model = model_cls(source=list(sources), **kwargs)
    data = [(u, v, attr) for u, v, attr in dataset.iter_edge_arrive(case_name)]
    processor = tqdm(iterable=data, desc=str(model_cls), total=len(data), unit='it')
    for u, v, attr in processor:
        model.edge_arrive(u, v, attr)
        process = (processor.n + 1) / processor.total
        current_speed = processor.format_dict["rate"]
        current_speed = (1 / current_speed)
        current_speed *= (len(data) / transaction_cnt)
        if current_speed is None:
            continue
        if record_points[point_idx] <= process:
            results.append(current_speed)
            point_idx += 1
        if current_speed < 1:
            return results + [1 for _ in range(len(record_points) - point_idx)]

    return results


def eval_tps_from_transaction_arrive(
        dataset: DynamicTransNetwork,
        case_name: str,
        model_cls: Any,
        record_points: List[float],
        **kwargs,
) -> List:
    """
    Perform the evaluation with the transaction arrived method.

    :param dataset: the dynamic network
    :param case_name: the tracing case name
    :param model_cls: the class of the transaction arrived method
    :return: the collected evaluating metrics
    """
    results = list()

    # init the source
    sources = set()
    addr2label = dataset.get_case_labels(case_name)
    for addr, label in addr2label.items():
        if label == 'ml_transit_0':
            sources.add(addr)

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
    point_idx = 0
    model = model_cls(source=list(sources), **kwargs)
    processor = tqdm(
        iterable=txhash_sorted,
        desc=str(model_cls),
        total=len(txhash_sorted),
        unit='it'
    )
    for txhash in processor:
        process = (processor.n + 1) / processor.total
        current_speed = processor.format_dict["rate"]
        if current_speed is None:
            continue
        if record_points[point_idx] <= process:
            results.append(current_speed)
            point_idx += 1
        if current_speed < 1:
            return results + [1 for _ in range(len(record_points) - point_idx)]

        # run the model
        trans = nx.MultiDiGraph()
        trans.add_edges_from(trans2edges[txhash])
        model.transaction_arrive(trans)
    return results


def eval_methods(
        dataset: DynamicTransNetwork,
        case_name: str,
        record_points: List[float],
) -> Dict:
    results = dict()
    transaction_cnt = dataset.get_case_transaction_count(case_name)
    # results['DTTR'] = eval_tps_from_transaction_arrive(
    #     dataset=dataset,
    #     case_name=case_name,
    #     model_cls=DTTR,
    #     record_points=record_points,
    # )
    results['DAPPR'] = eval_tps_from_edge_arrive(
        dataset=dataset,
        case_name=case_name,
        model_cls=DAPPR,
        record_points=record_points,
        transaction_cnt=transaction_cnt,
    )
    results['TILES'] = eval_tps_from_edge_arrive(
        dataset=dataset,
        case_name=case_name,
        model_cls=TILES,
        record_points=record_points,
        transaction_cnt=transaction_cnt,
    )
    for model_cls in [
        BFS, Poison, Haircut,
        APPR, TTRRedirect
    ]:
        results[model_cls.__name__] = eval_tps_from_pushpop(
            dataset=dataset,
            case_name=case_name,
            model_cls=model_cls,
            record_points=record_points,
            transaction_cnt=transaction_cnt,
        )

    # save the results to cache
    cache_fn = os.path.join(PROJECT_PATH, 'cache', 'time_cost.csv')
    with open(cache_fn, 'w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f)
        for method, tps in results.items():
            writer.writerow([method] + tps)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    parser.add_argument('--case', type=str, default='PlusTokenPonzi')
    args = parser.parse_args()
    dataset = DynamicTransNetwork(raw_path=args.raw_path)
    record_points = [0.05 * i for i in range(1, 20 + 1)]

    # load the cached results
    method2tps = dict()
    cache_fn = os.path.join(PROJECT_PATH, 'cache', 'time_cost.csv')
    if not os.path.exists(cache_fn):
        results = eval_methods(dataset, args.case, record_points)
    with open(cache_fn, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            method2tps[row[0]] = row[1:]

    # print the results
    for method, tps in method2tps.items():
        print(f"{method}: {', '.join(tps)}")
