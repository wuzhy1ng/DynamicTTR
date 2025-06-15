import argparse
import re

from dataset.dynamic import DynamicTransNetwork

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    args = parser.parse_args()
    dataset = DynamicTransNetwork(raw_path=args.raw_path)

    # Collect source and transit addresses
    sources = set()
    transits = set()
    services = set()
    pattern = r"ml_transit_.*?"
    for case_name in dataset.get_case_names():
        addr2label = dataset.get_case_labels(case_name)
        for addr, label in addr2label.items():
            if label == 'ml_transit_0':
                sources.add(addr)
            elif re.match(pattern, str(label)):
                transits.add(addr)
            else:
                services.add(addr)


    # Count token transfers and unique token contracts
    token_transfer_cnt = 0
    token_contracts = set()
    for case_name in dataset.get_case_names():
        for addr_from, addr_to, edge_data in dataset.iter_edge_arrive(case_name):
            token_transfer_cnt += 1
            token_contracts.add(edge_data['contractAddress'])

    print('Total number of sources:', len(sources))
    print('Total number of transits:', len(transits))
    print('Total number of services / others:', len(services))
    print('Total number of token transfers:', token_transfer_cnt)
    print('Total number of unique token contracts:', len(token_contracts))
