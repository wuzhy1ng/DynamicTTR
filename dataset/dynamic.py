import csv
import hashlib
import os
from typing import List, Iterator, Dict


class DynamicTransNetwork:
    def __init__(self, raw_path: str):
        self.raw_path = raw_path

    def get_case_names(self) -> List[str]:
        case_names = list()
        for name in os.listdir(self.raw_path):
            p = os.path.join(self.raw_path, name)
            if not os.path.isdir(p):
                continue
            case_names.append(name)
        return case_names

    def iter_edge_arrive(self, case_name: str) -> Iterator[tuple[str, str, Dict]]:
        path = os.path.join(self.raw_path, case_name, 'all-tx.csv')
        data = list()
        _ids = set()
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                _id = '{}_{}_{}_{}_{}'.format(
                    row['hash'], row['from'], row['to'],
                    row['tokenSymbol'], int(float(row['value']))
                )
                _id = _id.encode('utf-8')
                _id = hashlib.sha1(_id).hexdigest()
                if _id in _ids:
                    continue
                _ids.add(_id)
                data.append(row)
        data.sort(key=lambda x: int(x['timeStamp']))
        for row_data in data:
            contract_address = row_data['contractAddress'] \
                if row_data['contractAddress'] != '' \
                else '0x' + '0' * 40
            yield row_data['from'], row_data['to'], {
                'hash': row_data['hash'],
                'value': row_data['value'],
                'timeStamp': int(row_data['timeStamp']),
                'symbol': row_data['tokenSymbol'],
                'contractAddress': contract_address,
            }

    def get_case_labels(self, case_name: str) -> Dict[str, str]:
        result = dict()
        path = os.path.join(self.raw_path, case_name, 'all-address.csv')
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['name_tag'] == '':
                    continue
                result[row['address']] = row['name_tag']
        path = os.path.join(self.raw_path, case_name, 'accounts-hacker.csv')
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row['name_tag'] == '':
                    continue
                result[row['address']] = row['name_tag']
        return result

    def get_case_transaction_count(self, case_name: str) -> int:
        txhashes = set()
        path = os.path.join(self.raw_path, case_name, 'all-tx.csv')
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                txhashes.add(row['hash'])
        return len(txhashes)
