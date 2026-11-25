import os
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from examples.association.baselines import HEURISTICS, result_path
from examples.association.env import get_deployment, make_network, option_table, random_assoc, score
from examples.association.graph import build_topology
from examples.association.pipeline import (
    create_contexts,
    create_decoder,
    create_refiner,
    create_sampler,
    create_scorer,
    load_pipeline
)
from fm4mac.graphs import take_graphs
from fm4mac.metrics import device_metrics, gm_ratio


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--run', type=str, default='runs/flow_matching')
    args.add_argument('--surrogate', type=str, default=None)
    args.add_argument('--deployment', type=str, default='random/8')
    args.add_argument('--init', type=str, default='fm', choices=['fm', 'heuristic', 'random'])
    args.add_argument('--n_contexts', type=int, default=4)
    args.add_argument('--n_samples', type=int, default=128)
    args.add_argument('--top_k', type=int, default=4)
    args.add_argument('--n_rounds', type=int, default=2)
    args.add_argument('--eval_seeds', type=int, default=16)
    args.add_argument('--label', type=str, default='zero_shot')
    args = args.parse_args()

    fm, params, encode_fn, decode_fn, surrogate, s_params = load_pipeline(args.run, args.surrogate)
    deployment = get_deployment(args.deployment)
    net, topology, primary = make_network(deployment), build_topology(deployment), jnp.asarray(deployment.primary)
    sample, decode = create_sampler(fm), create_decoder(decode_fn, topology)
    score_fn = create_scorer(surrogate, topology, primary)
    refine = create_refiner(score_fn)
    table = option_table(net)
    path = result_path(args.deployment, 'baselines')
    baselines = json.load(open(path)) if args.init == 'heuristic' or os.path.exists(path) else None
    rng = np.random.default_rng(0)
    top1, refined, times = [], [], []

    for key in jax.random.split(jax.random.PRNGKey(42), args.n_contexts):
        start = time.time()

        if args.init == 'fm':
            context_key, sample_key, plan_key = jax.random.split(key, 3)
            contexts = create_contexts(context_key, net, topology, primary, encode_fn, 1)
            contexts = take_graphs(contexts, np.zeros(args.n_samples, dtype=int))
            z = sample(params, sample_key, contexts)
            plans = np.asarray(decode(jax.random.split(plan_key, args.n_samples), contexts, z))
        elif args.init == 'random':
            plans = np.stack([random_assoc(rng, net) for _ in range(args.n_samples)])
        else:
            plans = np.concatenate([np.asarray(baselines['plans'][name]) for name in HEURISTICS])

        order = np.argsort(-np.asarray(score_fn(s_params, jnp.asarray(plans))))
        top1.append(plans[order[0]])
        initial = plans[order[:args.top_k]]
        found, values = refine(s_params, initial, np.maximum(table, 0), table >= 0, rng, args.n_rounds)
        refined.append(found[np.argmax(values)])
        times.append(time.time() - start)

    eval_key = jax.random.PRNGKey(7)
    results = {
        f'{args.init}_top1': device_metrics(score(eval_key, net, np.stack(top1), args.eval_seeds)[1]),
        f'{args.init}_refined': device_metrics(score(eval_key, net, np.stack(refined), args.eval_seeds)[1])
    }

    if baselines is not None:
        results |= baselines['results']
        ls = baselines['results']['local_search']['pf']
        best = max(HEURISTICS, key=lambda name: baselines['results'][name]['pf'])
        names = [*results][:2] + ['strongest', best]
        ratios = {name: gm_ratio(results[name]['pf'], ls, topology.n_stas) for name in names}
        print('geometric-mean throughput relative to local search', {k: round(float(v), 4) for k, v in ratios.items()})

    for name, m in results.items():
        print(f'{name:22s}', {k: round(v, 2) for k, v in m.items()})

    print('inference time per context [s]', round(float(np.mean(times)), 1))
    os.makedirs(os.path.dirname(result_path(args.deployment, args.label)), exist_ok=True)

    with open(result_path(args.deployment, args.label), 'w') as f:
        plans = dict(top1=np.stack(top1).tolist(), refined=np.stack(refined).tolist())
        json.dump(dict(results=results, time=float(np.mean(times)), plans=plans, args=vars(args)), f)
