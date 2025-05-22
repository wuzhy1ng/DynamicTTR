import argparse
import csv
import os.path

from algos.dttr import DTTR
from comp import eval_method, eval_case_from_transaction_arrive
from dataset.dynamic import DynamicTransNetwork
from settings import PROJECT_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, required=True)
    args = parser.parse_args()

    dataset = DynamicTransNetwork(raw_path=args.raw_path)
    cached_result_path = os.path.join(PROJECT_PATH, 'cache', 'sens_exps.csv')
    file = open(cached_result_path, 'w', encoding='utf-8', newline='\n')
    writer = csv.writer(file)
    writer.writerow(['alpha', 'epsilon', 'depth', 'recall', 'precision', 'fpr', 'size', 'tps'])
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25]:
        for epsilon in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
            print('alpha:', alpha, 'epsilon:', epsilon)
            avg_depth, avg_recall, avg_precision, avg_fpr, size, tps = eval_method(
                dataset=dataset,
                model_cls=DTTR,
                eval_fn=eval_case_from_transaction_arrive,
                alpha=alpha,
                epsilon=epsilon,
            )
            print(
                alpha, epsilon, avg_depth,
                avg_recall, avg_precision, avg_fpr,
                size, tps,
            )
            writer.writerow([
                alpha, epsilon, avg_depth,
                avg_recall, avg_precision, avg_fpr,
                size, tps,
            ])
    file.close()
