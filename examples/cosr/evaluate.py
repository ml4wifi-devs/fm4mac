import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
import time
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import numpy as np

from examples.cosr.env import (
    T_OPTIMAL,
    create_simulator,
    designated_links,
    get_scenario,
    random_choice,
    round_robin_designated
)
from examples.cosr.graph import choice_to_arrays, create_structure
from examples.cosr.pipeline import Pipeline, default_surrogate, path
from fm4mac.graphs import take_graphs
from fm4mac.search import rank_correlation, select_covering


def create_env(pipe, scenario, n_designated, cfg, sim_chunk_size):
    """
    Evaluation set of a deployment: `cfg.n_contexts` observations, each repeated for `cfg.n_samples` candidates with
    designated stations that visit all stations in turn.
    """

    st = create_structure(scenario)
    contexts = pipe.contexts(jax.random.PRNGKey(cfg.seed), scenario, st, cfg.n_contexts)
    designated = jnp.concatenate([
        round_robin_designated(k, scenario.associations, n_designated, cfg.n_samples)
        for k in jax.random.split(jax.random.PRNGKey(cfg.seed + 1), cfg.n_contexts)
    ])
    return {
        'scenario': scenario, 'st': st, 'n_designated': n_designated, 'designated': designated,
        'simulate': create_simulator(scenario, st, sim_chunk_size),
        'contexts': take_graphs(contexts, np.repeat(np.arange(cfg.n_contexts), cfg.n_samples))
    }


def rollout(pipe, key, params, env, contexts, designated, n_sim_repeats=1):
    """
    Generates one candidate per context and simulates it. Returns latents, decisions, data rates and link rates.
    """

    sample_key, tx_key, sim_key = jax.random.split(key, 3)
    z, *decisions = pipe.generate(sample_key, tx_key, params, env['st'], contexts, designated)
    thr, rates = zip(*[env['simulate'](k, *decisions) for k in jax.random.split(sim_key, n_sim_repeats)])
    return z, decisions, np.mean(thr, axis=0), np.mean(rates, axis=0)


def evaluate(key, pipe, params, s_params, env, cfg, refine):
    """
    Top-k of `cfg.n_samples` candidates per observation selected by the surrogate (covering the designated stations),
    each simulated `cfg.n_repeats` times, optionally after refinement of the top `cfg.refine_top_k`. Candidates are
    FM samples or random decisions (`cfg.candidates`), scores surrogate predictions or random (`cfg.scores`).
    """

    rollout_key, sim_key = jax.random.split(key)
    st, simulate, designated = env['st'], env['simulate'], np.asarray(env['designated'])

    def resimulate(decisions, idxs):
        repeated = [np.repeat(a[idxs], cfg.n_repeats, axis=0) for a in decisions]
        return simulate(sim_key, *repeated)[0].reshape(len(idxs), cfg.n_repeats)

    if cfg.candidates == 'random':
        choice, power, mcs = random_choice(np.random.default_rng(cfg.seed), st, len(designated))
        fixed = designated_links(st, designated)
        decisions = (np.where(fixed >= 0, fixed, choice), power, mcs)
        rewards = simulate(jax.random.split(rollout_key, 3)[2], *decisions)[0]
    else:
        _, decisions, rewards, _ = rollout(pipe, rollout_key, params, env, env['contexts'], env['designated'])

    rewards = rewards.reshape(-1, cfg.n_samples)

    if cfg.scores == 'random':
        scores = np.random.default_rng(cfg.seed).random(rewards.shape)
    else:
        scores = pipe.scores(s_params, st, *decisions).reshape(rewards.shape)

    served = np.asarray(choice_to_arrays(st, *map(jnp.asarray, decisions))[0], dtype=bool)
    served = served.reshape(*scores.shape, -1)
    cover = (designated[..., 1, None] == st.link_sta).any(axis=-2).reshape(served.shape)
    offsets = cfg.n_samples * np.arange(rewards.shape[0])[:, None]
    metrics = {'sample_mean': rewards.mean().item(), 'surrogate_rank_corr': rank_correlation(scores, rewards)}
    selected = {}

    for top_k in cfg.top_ks:
        idxs = np.stack([select_covering(s, c, top_k, cfg.coverage_ratio) for s, c in zip(scores, cover)])
        selected[top_k] = (idxs + offsets).reshape(-1)
        metrics[f'top_{top_k}'] = resimulate(decisions, selected[top_k]).mean().item()
        metrics[f'top_{top_k}_coverage'] = np.take_along_axis(served, idxs[..., None], axis=1).any(axis=1).mean().item()

    if refine:
        top_k, idxs = cfg.refine_top_k, selected[cfg.refine_top_k]
        fixed = designated_links(st, designated[idxs])
        start = time.time()
        *refined, refined_scores = pipe.refine(
            s_params, st, *(a[idxs] for a in decisions), fixed, np.random.default_rng(cfg.refine_seed), cfg.n_rounds
        )
        refine_time = time.time() - start
        rates = resimulate(refined, np.arange(len(idxs))).reshape(-1, top_k, cfg.n_repeats)
        refined_served = np.asarray(choice_to_arrays(st, *map(jnp.asarray, refined))[0], dtype=bool)
        refined_served = refined_served.reshape(-1, top_k, served.shape[-1])

        metrics['top_1_refined'] = rates[:, 0].mean().item()
        metrics[f'top_{top_k}_refined'] = rates.mean().item()
        metrics['top_1_refined_coverage'] = refined_served[:, 0].mean().item()
        metrics[f'top_{top_k}_refined_coverage'] = refined_served.any(axis=1).mean().item()
        metrics[f'top_{top_k}_refined_score_gain'] = (refined_scores - scores.reshape(-1)[idxs]).mean().item()
        metrics['refine_designated_kept'] = float(((fixed < 0) | (refined[0] == fixed)).all())
        metrics['refine_time_per_context'] = refine_time / rates.shape[0]

    return metrics


if __name__ == '__main__':
    args = ArgumentParser(description='Zero-shot (or adapted, with an RL run as the checkpoint) Co-SR evaluation')
    args.add_argument('--scenario', type=str, default='residential/3')
    args.add_argument('--run', type=str, default='runs/flow_matching', help='Pretrained FM or RL run')
    args.add_argument('--surrogate', type=str, default=None, help='Default: RL run or runs/surrogate')
    args.add_argument('--n_designated', type=int, default=3)
    args.add_argument('--coverage_ratio', type=float, default=1.)
    args.add_argument('--candidates', type=str, choices=['fm', 'random'], default='fm')
    args.add_argument('--scores', type=str, choices=['surrogate', 'random'], default='surrogate')
    args.add_argument('--no_refine', action='store_true', default=False)
    args.add_argument('--n_contexts', type=int, default=32)
    args.add_argument('--n_samples', type=int, default=128)
    args.add_argument('--n_repeats', type=int, default=5)
    args.add_argument('--top_ks', type=int, nargs='+', default=[1, 8, 32, 128])
    args.add_argument('--refine_top_k', type=int, default=8)
    args.add_argument('--n_rounds', type=int, default=2)
    args.add_argument('--refine_seed', type=int, default=0)
    args.add_argument('--refine_batch_size', type=int, default=256)
    args.add_argument('--chunk_size', type=int, default=64)
    args.add_argument('--sim_chunk_size', type=int, default=2048)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--output', type=str, default=None)
    args = args.parse_args()

    pipe = Pipeline(
        args.run, args.surrogate or default_surrogate(args.run), chunk_size=args.chunk_size,
        refine_batch_size=args.refine_batch_size
    )
    scenario = get_scenario(args.scenario)
    env = create_env(pipe, scenario, args.n_designated, args, args.sim_chunk_size)
    metrics = evaluate(jax.random.PRNGKey(args.seed), pipe, pipe.params, pipe.s_params, env, args, not args.no_refine)

    split, idx = args.scenario.split('/')
    if split in T_OPTIMAL:
        t_opt = T_OPTIMAL[split][int(idx)]
        metrics |= {f'{k}_rel': metrics[k] / t_opt for k in ('top_1', 'top_8', 'top_8_refined') if k in metrics}

    print(json.dumps(metrics, indent=1))

    if args.output is not None:
        os.makedirs(os.path.dirname(path(args.output)), exist_ok=True)
        with open(path(args.output), 'w') as f:
            json.dump({'args': vars(args), 'metrics': metrics}, f, indent=1)
