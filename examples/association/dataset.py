import os
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
from dataclasses import dataclass

import jax
import numpy as np
from tqdm import tqdm

from examples.association.baselines import (
    AirtimeModel,
    create_local_search,
    estimated_throughput,
    model_pf,
    strongest_signal
)
from examples.association.env import (
    TRAIN_PARAMS,
    Deployment,
    draw_deployment,
    make_network,
    option_table,
    random_assoc,
    score
)
from fm4mac.data import save_dataset


ROOT = os.path.dirname(os.path.abspath(__file__))
RANDOM, STRONGEST, ESTIMATED, MODEL_PF, LOCAL_SEARCH = range(5)


@dataclass
class DeploymentItem:
    deployment: Deployment
    plans: np.ndarray
    throughput: np.ndarray
    value: np.ndarray
    source: np.ndarray


def generate_item(key, rng, n_random=32, n_seeds=8, n_rounds=2, top_k=16):
    """
    Associations of one random deployment with their simulated per-STA throughput: random associations, the
    strongest-signal, estimated-throughput and airtime-model PF associations, and the best associations visited by
    local search started from the best of these heuristics.
    """

    eval_key, init_key, ls_key = jax.random.split(key, 3)
    deployment = draw_deployment(rng)
    net = make_network(deployment)
    model = AirtimeModel(net, TRAIN_PARAMS)
    ssf = strongest_signal(net)

    heuristics = [ssf, estimated_throughput(model, ssf, rng), model_pf(model, ssf, rng)]
    plans = [random_assoc(rng, net) for _ in range(n_random)] + heuristics
    source = [RANDOM] * n_random + [STRONGEST, ESTIMATED, MODEL_PF]

    init = heuristics[int(np.argmax(score(init_key, net, np.stack(heuristics), n_seeds)[0]))]
    table = option_table(net)
    *_, history = create_local_search(n_seeds)(
        (net, ls_key), init[None], np.maximum(table, 0), table >= 0, rng, n_rounds, visited=True
    )
    visited = {tuple(p): v for candidates, scores in history for p, v in zip(candidates[0], scores[0]) if v > -np.inf}
    best = sorted(visited, key=visited.get)[-top_k:]
    plans += [np.array(p) for p in best]
    source += [LOCAL_SEARCH] * len(best)

    plans = np.stack(plans)
    value, thr = score(eval_key, net, plans, n_seeds)
    return DeploymentItem(deployment, plans, thr, value, np.array(source))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--n_deployments', type=int, default=100)
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--output', type=str, default='datasets/train_0.pkl.lz4')
    args = args.parse_args()

    rng = np.random.default_rng(args.seed)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_deployments)
    items = [generate_item(key, rng) for key in tqdm(keys)]
    output = os.path.join(ROOT, args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    save_dataset(items, output)
