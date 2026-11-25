import os
os.environ['JAX_ENABLE_X64'] = 'True'

from argparse import ArgumentParser

import jax
import numpy as np
from tqdm import tqdm

from examples.channel.baselines import coloring_plan, local_search
from examples.channel.env import random_deployment, random_plan, score
from fm4mac.data import save_dataset


ROOT = os.path.dirname(os.path.abspath(__file__))
RANDOM, COLORING, LOCAL_SEARCH = 0, 1, 2


def draw_deployment(rng, key, n_aps_range=(2, 12), n_stas_range=(1, 4), ap_distance_range=(5., 15.)):
    n_aps = int(rng.integers(n_aps_range[0], n_aps_range[1] + 1))
    n_stas = int(rng.integers(n_stas_range[0], n_stas_range[1] + 1))
    ap_pos, sta_pos = random_deployment(key, n_aps, n_stas, float(rng.uniform(*ap_distance_range)))
    return np.asarray(ap_pos), np.asarray(sta_pos), np.ones((n_aps, n_stas), dtype=bool)


def generate_item(key, rng, n_random=32, n_seeds=2, n_rounds=2, top_k=16, kind='pf'):
    """
    Plans of one random deployment with their simulated per-AP throughput: random plans, colouring plans and the
    best distinct plans visited by local search started from the colouring.
    """

    dep_key, rand_key, eval_key, ls_key = jax.random.split(key, 4)
    deployment = draw_deployment(rng, dep_key)
    ap_pos, sta_pos, _ = deployment

    plans = [random_plan(k, len(ap_pos)) for k in jax.random.split(rand_key, n_random)]
    plans += [coloring_plan(ap_pos, sta_pos, strict=strict, rng=rng) for strict in (True, False)]
    source = [RANDOM] * n_random + [COLORING] * 2

    _, history = local_search(ls_key, deployment, plans[-1][None], kind, n_rounds, n_seeds, rng, visited=True)
    visited = np.concatenate([candidates[0] for candidates, _ in history])
    values = np.concatenate([values[0] for _, values in history])
    unique, idxs = np.unique(visited, axis=0, return_index=True)
    best = unique[np.argsort(values[idxs])[-top_k:]]
    plans += list(best)
    source += [LOCAL_SEARCH] * len(best)

    plans = np.stack(plans)
    _, thr = score(eval_key, deployment, plans, n_seeds, kind)
    return dict(
        ap_pos=ap_pos, sta_pos=sta_pos, sta_mask=deployment[2], plans=plans, throughput=thr, source=np.array(source)
    )


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--n_deployments', type=int, default=100)
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--objective', type=str, default='pf', choices=['pf', 'sum'])
    args.add_argument('--output', type=str, default='datasets/train_0.pkl.lz4')
    args = args.parse_args()

    rng = np.random.default_rng(args.seed)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_deployments)
    items = [generate_item(key, rng, kind=args.objective) for key in tqdm(keys)]

    output = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_dataset(items, output)
