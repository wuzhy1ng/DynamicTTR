import argparse
import csv
import os.path

from algos.dttr import DTTR
from comp import eval_method, eval_case_from_transaction_arrive, eval_case_from_edge_arrive
from dataset.dynamic import DynamicTransNetwork
from settings import PROJECT_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    args = parser.parse_args()

    dataset = DynamicTransNetwork(raw_path=args.raw_path)
    cached_result_path = os.path.join(PROJECT_PATH, 'cache', 'abl_exps.csv')
    file = open(cached_result_path, 'w', encoding='utf-8', newline='\n')
    writer = csv.writer(file)
    writer.writerow(['type', 'depth', 'recall', 'precision', 'fpr', 'size', 'tps'])

    # w/o log weight
    avg_depth, avg_recall, avg_precision, avg_fpr, size, tps = eval_method(
        dataset=dataset,
        model_cls=DTTR,
        eval_fn=eval_case_from_transaction_arrive,
        is_log_value=False,
    )
    writer.writerow([
        'w/o log weight', avg_depth,
        avg_recall, avg_precision, avg_fpr,
        size, tps,
    ])

    # w/o log weight and swap reduction
    avg_depth, avg_recall, avg_precision, avg_fpr, size, tps = eval_method(
        dataset=dataset,
        model_cls=DTTR,
        eval_fn=eval_case_from_transaction_arrive,
        is_reduce_swap=False,
        is_log_value=False,
    )
    writer.writerow([
        'w/o log weight and swap reduction', avg_depth,
        avg_recall, avg_precision, avg_fpr,
        size, tps,
    ])

    # w/o log weight, swap reduction, and pricing
    avg_depth, avg_recall, avg_precision, avg_fpr, size, tps = eval_method(
        dataset=dataset,
        model_cls=DTTR,
        eval_fn=eval_case_from_transaction_arrive,
        is_in_usd=False,
        is_reduce_swap=False,
        is_log_value=False,
    )
    writer.writerow([
        'w/o log weight, swap reduction, and pricing', avg_depth,
        avg_recall, avg_precision, avg_fpr,
        size, tps,
    ])
    file.close()
