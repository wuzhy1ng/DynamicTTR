import os
from typing import List, Iterator, Dict

import pandas as pd


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
        data = pd.read_csv(path)
        data.sort_values(by=['timeStamp'], inplace=True)
        for _, row in data.iterrows():
            row_data = row.to_dict()
            yield row_data['from'], row_data['to'], {
                'value': row_data['value'],
                'timeStamp': row_data['timeStamp'],
                'symbol': row_data['tokenSymbol'],
            }

    def get_case_labels(self, case_name: str) -> Dict[str, str]:
        result = dict()
        path = os.path.join(self.raw_path, case_name, 'all-address.csv')
        data = pd.read_csv(path)
        for _, row in data.iterrows():
            row_data = row.to_dict()
            result[row_data['address']] = row_data['name_tag']
        return result
