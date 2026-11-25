import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.channel.baselines import coloring_plan, evaluate_plans
from examples.channel.env import get_deployment, random_plan
from examples.channel.graph import build_topology
from examples.channel.pipeline import (
    ROOT,
    create_refiner,
    create_sampler,
    create_scorer,
    load_autoencoder,
    load_flow_matching,
    load_surrogate,
    observe,
    resolve
)
from fm4mac.checkpoints import load_config
from fm4mac.graphs import take_graphs
from fm4mac.metrics import gm_ratio


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--run', type=str, default='runs/flow_matching')
    args.add_argument('--surrogate', type=str, default=None, help='Default: RL run or runs/surrogate')
    args.add_argument('--deployment', type=str, default='random/8')
    args.add_argument('--init', type=str, default='fm', choices=['fm', 'heuristic', 'random'])
    args.add_argument('--n_contexts', type=int, default=4)
    args.add_argument('--n_samples', type=int, default=128)
    args.add_argument('--top_k', type=int, default=4)
    args.add_argument('--n_rounds', type=int, default=2)
    args.add_argument('--chunk_size', type=int, default=128)
    args.add_argument('--eval_seeds', type=int, default=8)
    args.add_argument('--output_dir', type=str, default='results')
    args = args.parse_args()

    deployment = get_deployment(args.deployment)
    ap_pos, sta_pos, _ = deployment
    topology = build_topology(*deployment)
    n_aps = topology.n_aps

    fm, fm_params, autoencoder_path = load_flow_matching(args.run)
    encode_fn, decode_fn = load_autoencoder(autoencoder_path)
    rl_run = 'surrogate_path' in load_config(resolve(args.run))
    surrogate, s_params = load_surrogate(args.surrogate or (args.run if rl_run else 'runs/surrogate'))
    sample = create_sampler(fm, decode_fn, n_aps, args.chunk_size)
    score = create_scorer(surrogate, topology, args.chunk_size)
    refine = create_refiner(score, n_aps)
    scorer = lambda plans: np.asarray(score(s_params, jnp.asarray(plans)))
    rng = np.random.default_rng(0)
    top1, refined, times = [], [], []

    for key in jax.random.split(jax.random.PRNGKey(42), args.n_contexts):
        start = time.time()
        obs_key, gen_key, plan_key = jax.random.split(key, 3)

        if args.init == 'fm':
            contexts = observe(obs_key, deployment, topology, encode_fn, 1)
            _, plans = sample(fm_params, gen_key, plan_key, take_graphs(contexts, np.zeros(args.n_samples, dtype=int)))
            plans = np.asarray(plans)
        elif args.init == 'random':
            plans = np.stack([random_plan(k, n_aps) for k in jax.random.split(key, args.n_samples)])
        else:
            plans = np.stack([coloring_plan(ap_pos, sta_pos, strict=strict) for strict in (True, False)])

        order = np.argsort(-scorer(plans))
        top1.append(plans[order[0]])
        candidates, values = refine(s_params, plans[order[np.arange(args.top_k) % len(plans)]], rng, args.n_rounds)
        refined.append(candidates[np.argmax(values)])
        times.append(time.time() - start)

    name = args.deployment.replace('/', '_')
    results = {
        f'{args.init}_top1': evaluate_plans(deployment, np.stack(top1), args.eval_seeds),
        f'{args.init}_refined': evaluate_plans(deployment, np.stack(refined), args.eval_seeds)
    }
    baselines_path = os.path.join(ROOT, args.output_dir, f'{name}_baselines.json')

    if os.path.exists(baselines_path):
        with open(baselines_path) as f:
            reference = json.load(f)['results']['local_search_pf']['pf']

        for m in results.values():
            m['ratio'] = float(gm_ratio(m['pf'], reference, n_aps))

    for method, m in results.items():
        print(f'{method:16s}', {k: round(v, 3) for k, v in m.items()})

    print('inference time per context [s]', round(float(np.mean(times)), 1))
    os.makedirs(os.path.join(ROOT, args.output_dir), exist_ok=True)
    run_name = os.path.basename(os.path.normpath(args.run))

    with open(os.path.join(ROOT, args.output_dir, f'{name}_{run_name}_{args.init}.json'), 'w') as f:
        json.dump(dict(results=results, time=float(np.mean(times)), refined=np.stack(refined).tolist()), f, indent=1)
