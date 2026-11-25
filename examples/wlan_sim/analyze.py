import json
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from fm4mac.metrics import agreement, pf_utility


def mean_se(runs):
    return np.atleast_1d(runs.mean(0)), np.atleast_1d(runs.std(0, ddof=1) / np.sqrt(len(runs)))


def compare(results, level, select=lambda r: True, reduce=lambda x, r: x):
    """
    Agreement of the Komondor and JAX means over seeds of a per-`level` ('ap' or 'sta') quantity, `reduce`d per run.
    """

    stats = [
        (*mean_se(reduce(np.asarray(r[f'komondor_{level}']), r)), *mean_se(reduce(np.asarray(r[f'jax_{level}']), r)))
        for r in results if select(r)
    ]
    k, k_se, j, j_se = (np.concatenate(x) for x in zip(*stats))
    return agreement(k, j, k_se, j_se), k, j


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--input', type=str, default='examples/wlan_sim/validation/channel.json')
    args = args.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    results = data['results']

    out = {}
    out['per_ap'], k_ap, j_ap = compare(results, 'ap')
    out['per_sta'], k_sta, j_sta = compare(results, 'sta')
    if data['kind'] == 'channel':
        for name, policy in [('OP', 0), ('SCB', 1), ('AM', 2)]:
            select_policy = lambda x, r, p=policy: x[:, np.array(r['policy']) == p]
            out[f'per_ap_{name}'] = compare(results, 'ap', reduce=select_policy)[0]
    else:
        for name, strongest in [('strongest', True), ('random', False)]:
            out[f'per_sta_{name}'] = compare(results, 'sta', select=lambda r, s=strongest: r['strongest'] == s)[0]

    out['aggregate'], k_tot, j_tot = compare(results, 'ap', reduce=lambda x, r: x.sum(-1))
    out['aggregate']['spearman'] = float(spearmanr(k_tot, j_tot)[0])
    out['pf_utility'], k_pf, j_pf = compare(results, 'sta', reduce=lambda x, r: pf_utility(x))
    out['pf_utility']['spearman'] = float(spearmanr(k_pf, j_pf)[0])
    out['starved_sta'] = dict(
        komondor=int((k_sta < 1.).sum()), jax=int((j_sta < 1.).sum()), both=int(((k_sta < 1.) & (j_sta < 1.)).sum())
    )

    stem = os.path.splitext(args.input)[0]
    with open(f'{stem}_summary.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    panels = [
        (k_ap, j_ap, 'Per-AP throughput [Mbps]'), (k_sta, j_sta, 'Per-STA throughput [Mbps]'),
        (k_tot, j_tot, 'Aggregate throughput [Mbps]'), (k_pf, j_pf, 'PF utility (sum log Mbps)')
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (x, y, title) in zip(axes, panels):
        ax.scatter(x, y, s=5, alpha=0.6, color='#2a78d6')
        lim = [min(x.min(), y.min(), 0), max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, color='#52514e', lw=0.8, ls='--')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('Komondor v4.2.1')
        ax.set_ylabel('JAX simulator')
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f'{stem}_scatter.png', dpi=150)
