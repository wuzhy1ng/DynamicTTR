import csv
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
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                data.append(row)
        data.sort(key=lambda x: int(x['timeStamp']))
        for row_data in data:
            if row_data['isError'] != '':
                continue
            contract_address = row_data['contractAddress'] \
                if row_data['contractAddress'] != '' \
                else '0x' + '0' * 40
            yield row_data['from'], row_data['to'], {
                'value': row_data['value'],
                'timeStamp': int(row_data['timeStamp']),
                'contractAddress': contract_address,
            }

    def get_case_labels(self, case_name: str) -> Dict[str, str]:
        result = dict()
        path = os.path.join(self.raw_path, case_name, 'all-address.csv')
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                result[row['address']] = row['name_tag']
        return result
