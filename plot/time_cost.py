import argparse
import csv
import math
import os.path
import time
from typing import Any, List, Dict

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from algos.appr import APPR
from algos.bfs import BFS
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
        time_used = time.time()
        aggregator.execute(graph)
        time_used = time.time() - time_used

        current_speed = time_used + loading_time * (processor.n + 1) / len(data)
        current_speed = 1 / current_speed
        current_speed *= (transaction_cnt / len(data))
        if record_points[point_idx] <= processor.n + 1:
            results.append(current_speed)
            print(model_cls, record_points[point_idx], current_speed)
            point_idx += 1
        if point_idx >= len(record_points):
            break
        if current_speed < 1:
            return results + [1 for _ in range(len(record_points) - point_idx - 1)]
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
        process = processor.n + 1
        current_speed = processor.format_dict["rate"]
        if current_speed is None:
            continue
        current_speed *= (transaction_cnt / len(data))
        if record_points[point_idx] <= process:
            results.append(current_speed)
            print(model_cls, record_points[point_idx], current_speed)
            point_idx += 1
        if point_idx >= len(record_points):
            break
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
        trans = nx.MultiDiGraph()
        trans.add_edges_from(trans2edges[txhash])
        model.transaction_arrive(trans)
        process = processor.n + 1
        current_speed = processor.format_dict["rate"]
        if current_speed is None:
            continue
        if record_points[point_idx] <= process:
            results.append(current_speed)
            print(model_cls, record_points[point_idx], current_speed)
            point_idx += 1
        # if current_speed < 1:
        #     return results + [1 for _ in range(len(record_points) - point_idx)]

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
    # print(','.join(['DTTR', *[
    #     str(n) for n in results['DTTR']
    # ]]))
    #
    # results['LazyFwd'] = eval_tps_from_edge_arrive(
    #     dataset=dataset,
    #     case_name=case_name,
    #     model_cls=DAPPR,
    #     record_points=record_points,
    #     transaction_cnt=transaction_cnt,
    # )
    # print(','.join(['LazyFwd', *[
    #     str(n) for n in results['LazyFwd']
    # ]]))
    #
    # results['TILES'] = eval_tps_from_edge_arrive(
    #     dataset=dataset,
    #     case_name=case_name,
    #     model_cls=TILES,
    #     record_points=record_points,
    #     transaction_cnt=transaction_cnt,
    # )
    # print(','.join(['TILES', *[
    #     str(n) for n in results['TILES']
    # ]]))

    for model_cls in [
        # BFS, Poison, Haircut,
        # APPR, TTRRedirect,
        TPP, LPA, LOUVAIN,
    ]:
        results[model_cls.__name__] = eval_tps_from_pushpop(
            dataset=dataset,
            case_name=case_name,
            model_cls=model_cls,
            record_points=record_points,
            transaction_cnt=transaction_cnt,
        )
        print(','.join([model_cls.__name__, *[
            str(n) for n in results[model_cls.__name__]
        ]]))

    return results


def plot(record_points: List[float], transaction_cnt: int):
    method2tps = dict()
    fn = os.path.join(PROJECT_PATH, 'cache', 'time_cost.csv')
    with open(fn, 'r') as f:
        for row in csv.reader(f):
            nums = [float(n) for n in row[1:]]
            method2tps[row[0]] = nums

    # 定义子图布局
    fig, axes = plt.subplots(
        2, 6, figsize=(20, 5.5),
        sharex=True, sharey=True
    )
    axes = axes.flatten()  # 将子图数组展平
    fig.delaxes(axes[-1])  # 删除多余的第12个子图

    # 定义颜色
    blue_methods = ['TILES', 'LazyFwd', 'Ours']

    # 绘制每个方法的折线图
    for i, (method, tpss) in enumerate(method2tps.items()):
        if i >= 11:  # 确保只绘制11个图
            break
        ax = axes[i]
        if method in blue_methods:
            ax.plot(record_points, tpss, label=method, linewidth=5)
        else:
            ax.plot(record_points, tpss, label=method, color='gray', linewidth=2)
        ax.set_title(method, fontsize=22)
        # ax.set_ylim(1, 10**6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=22)
        ax.grid()

    # 只给最左侧子图添加y轴标签
    for i in [0, 6]:  # 第一列的行索引
        axes[i].set_ylabel("TPS", fontsize=22)
    # 只给最底部子图添加x轴标签
    for k in range(6, 11):  # 第二行的列索引
        axes[k].set_xlabel("#Transaction", fontsize=22)


    # 调整布局
    plt.tight_layout()
    plt.savefig('time_cost.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    parser.add_argument('--case', type=str, default='PlusTokenPonzi')
    args = parser.parse_args()
    dataset = DynamicTransNetwork(raw_path=args.raw_path)
    transaction_cnt = dataset.get_case_transaction_count(args.case)
    # record_points = [0.05 * i for i in range(1, 20 + 1)]
    record_points = [
        10 ** ((0.1 * i) * math.log10(transaction_cnt))
        for i in range(1, 10 + 1)
    ]
    record_points[-1] = transaction_cnt
    print('record_points:', record_points)

    # load the cached results
    # method2tps = dict()
    # results = eval_methods(dataset, args.case, record_points)

    # print the results
    plot(record_points, transaction_cnt)
