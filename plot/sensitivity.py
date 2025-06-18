import csv
from typing import Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt


def load_data(fn: str) -> Dict[Tuple, Dict]:
    rlt = dict()
    with open(fn, 'r') as f:
        for item in csv.DictReader(f):
            alpha = item.pop('alpha')
            epsilon = item.pop('epsilon')
            rlt[(alpha, epsilon)] = {
                'Recall': float(item['recall']) * 100,
                'Precision': float(item['precision']) * 100,
                'Depth': float(item['depth']),
                'TPS': float(item['tps']),
            }
    return rlt


def plot(data: Dict[Tuple, Dict]):
    metrics = ['Recall', 'Precision', 'Depth', 'TPS']
    alphas = sorted(set(key[0] for key in data.keys()), reverse=True)
    epsilons = sorted(set(key[1] for key in data.keys()))

    # 创建子图
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(20, 4.5),
    )
    fig.dpi = 768

    for i, metric in enumerate(metrics):
        # 构建热力图数据
        heatmap = np.zeros((len(alphas), len(epsilons)))
        for alpha_idx, alpha in enumerate(alphas):
            for epsilon_idx, epsilon in enumerate(epsilons):
                heatmap[alpha_idx, epsilon_idx] = float(data[(alpha, epsilon)][metric])

        # 绘制热力图
        ax = axes[i]
        im = ax.imshow(heatmap, aspect='auto', cmap='PuBu')
        ax.set_title(metric, fontsize=16)
        ax.set_xlabel(r'$\epsilon$', fontsize=16)
        ax.set_ylabel(r'$\alpha$', fontsize=16)
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels(epsilons, rotation=45, fontsize=16)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels(alphas, fontsize=16)

        # 在每个块中显示具体数值
        for alpha_idx in range(len(alphas)):
            for epsilon_idx in range(len(epsilons)):
                value = heatmap[alpha_idx, epsilon_idx]
                mid_val = (heatmap.min() + heatmap.max()) / 2
                val = f'{value:.0f}' if metric != 'Depth' else f'{value:.2f}'
                ax.text(
                    epsilon_idx, alpha_idx, val,
                    ha='center', va='center',
                    color='white' if value > mid_val else 'black',
                    fontsize=16,
                )

        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        # cbar.set_label(fontsize=16)

    plt.tight_layout()
    plt.savefig('sens.pdf')


if __name__ == '__main__':
    fn = r'C:\Users\87016\py_projects\DynamicTTR\cache\sens_exps.csv'
    data = load_data(fn)
    plot(data)
