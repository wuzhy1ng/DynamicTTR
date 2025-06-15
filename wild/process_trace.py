import csv
import json
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Dict

import requests
from future.backports.datetime import datetime


def query_addr_labels(address: str) -> List[str]:
    try:
        response = requests.post(
            url='http://localhost:55004/api/v2/get_address_label',
            json={
                "jsonrpc": "2.0",
                "id": "0",
                "method": "get_address_label",
                "params": {"addresses": [address]},
            }
        )
        data = response.json()
        return data['result'][address]
    except:
        return []


def load_gephi_nodes(trace_fn: str) -> List[Dict]:
    addr2labels = dict()
    with open(trace_fn, 'r') as f:
        data = json.load(f)
        for item in data:
            addr2labels[item['source']] = item['label'][0]
            for addr in item['distribution'].keys():
                addr2labels[addr] = addr2labels.get(addr, None)

    # coll label from api
    batch_size = 8
    executor = ThreadPoolExecutor(max_workers=batch_size)
    unlabelled_addrs = [addr for addr, label in addr2labels.items() if label is None]
    while len(unlabelled_addrs) > batch_size:
        batch_addrs = unlabelled_addrs[:batch_size]
        rlts = executor.map(query_addr_labels, batch_addrs)
        for addr, labels in zip(batch_addrs, rlts):
            addr2labels[addr] = ','.join(labels)
            print('get labels for address:', addr, 'labels:', labels)
        unlabelled_addrs = unlabelled_addrs[batch_size:]
        print(datetime.now(), 'left unlabelled addresses:', len(unlabelled_addrs))
    return [
        {'Id': addr, 'label': labels}
        for addr, labels in addr2labels.items()
    ]


def load_gephi_edges(trace_fn: str) -> List[Dict]:
    results = list()
    with open(trace_fn, 'r') as f:
        data = json.load(f)
        for item in data:
            for key, value in item['distribution'].items():
                results.append({
                    'Source': item['source'],
                    'Target': key,
                    'Weight': value,
                })
    return results


if __name__ == '__main__':
    trace_fn = './wild_trace.json'

    edges = load_gephi_edges(trace_fn)
    with open('./edges.csv', 'w', encoding='utf-8', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=['Source', 'Target', 'Weight'])
        writer.writeheader()
        for edge in edges:
            writer.writerow(edge)

    nodes = load_gephi_nodes(trace_fn)
    with open('./nodes.csv', 'w', encoding='utf-8', newline='\n') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'label'])
        writer.writeheader()
        for node in nodes:
            writer.writerow(node)
