import os
os.environ['JAX_ENABLE_X64'] = 'True'

import copy
import json
import pickle
import time
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from examples.cosr.baselines import create_agent, form_groups, rssi_matrix
from examples.cosr.env import (
    batch_data_rate_fn,
    designated_links,
    round_robin_designated,
    sample_designated,
    simulate_schedule
)
from examples.cosr.graph import choice_to_arrays, create_structure, designated_edges, selected_to_tx
from examples.cosr.mobility import add_env_args, create_env
from examples.cosr.pipeline import Pipeline, create_nft_step, create_surrogate_step, path
from fm4mac.graphs import take_graphs
from fm4mac.rl import group_advantages, nft_rewards, old_params_update
from fm4mac.search import select_covering
from fm4mac.training import create_optimizer


def build_trajectory(env, n_steps, scenario_seed, spawn_until_steps, enterprise):
    """
    Positions, simulation keys and associations (at every topology change) of a mobility trajectory, independent of
    the method.
    """

    key = env.reset(jax.random.PRNGKey(scenario_seed))
    pos, eval_keys, assoc_at = [], [], {}

    with jax.default_device(jax.devices('cpu')[0]):
        for step in trange(n_steps, desc='Trajectory'):
            if enterprise and spawn_until_steps is not None and step >= spawn_until_steps:
                env.spawn_count = 0

            key, _, info = env.step(key, step)
            key, eval_key = jax.random.split(key)
            pos.append(np.array(env.state.pos))
            eval_keys.append(np.asarray(eval_key))

            if step == 0 or info.get('topology_changed', False):
                assoc_at[step] = {int(ap): [int(s) for s in stas] for ap, stas in env.scenario.associations.items()}

    return {'pos': pos, 'eval_keys': np.stack(eval_keys), 'assoc_at': assoc_at, 'change_steps': sorted(assoc_at)}


def state_scenario(base, traj, step, enterprise):
    """
    Copy of the scenario at the positions and associations of `step`.
    """

    scenario = copy.copy(base)
    pos = traj['pos'][step]
    scenario.pos = jnp.asarray(pos)
    scenario.associations = traj['assoc_at'][max(s for s in traj['change_steps'] if s <= step)]

    if enterprise:
        scenario.walls = jnp.zeros((len(pos), len(pos)))
        scenario.tx_power = jnp.full(len(pos), base.tx_power[0].item())

    return scenario


class Evaluator:
    """
    Simulates per-step configurations (tx matrix, power and MCS indices) at the positions and keys of the trajectory, in
    batches of steps with the same number of nodes. Configurations generated before stations joined are padded (the new
    stations stay silent).
    """

    def __init__(self, base, traj, enterprise, chunk_size=1024):
        self.base, self.traj, self.enterprise, self.chunk_size = base, traj, enterprise, chunk_size
        self.tx_power = float(np.asarray(base.tx_power)[0])

    def __call__(self, steps, configs, oracle=False):
        steps = np.asarray(steps)
        n_nodes = np.asarray([self.traj['pos'][s].shape[0] for s in steps])
        thr, rates = np.zeros(len(steps)), [None] * len(steps)

        for n in np.unique(n_nodes):
            idxs = np.flatnonzero(n_nodes == n)
            walls = np.zeros((n, n)) if self.enterprise else np.asarray(self.base.walls)

            for i in range(0, len(idxs), self.chunk_size):
                part = idxs[i:i + self.chunk_size]
                sizes = [len(configs[j][1]) for j in part]
                tx = np.stack([np.pad(configs[j][0], (0, n - k)) for j, k in zip(part, sizes)])
                tx_power, mcs = (
                    np.stack([np.pad(configs[j][a], (0, n - k)) for j, k in zip(part, sizes)]) for a in (1, 2)
                )
                pos = np.stack([self.traj['pos'][s] for s in steps[part]])
                out_thr, out_rates = batch_data_rate_fn(
                    jnp.asarray(self.traj['eval_keys'][steps[part]]), tx,
                    self.tx_power - self.base.tx_power_delta * tx_power, mcs, pos, self.base.sigma, walls,
                    self.base.channel_width, self.base.path_loss_fn, oracle
                )
                thr[part] = np.asarray(out_thr)

                for j, k, r in zip(part, sizes, np.asarray(out_rates)):
                    rates[j] = r[:k]

        return thr, rates


def network_state(pipe, scenario, key, n_contexts):
    """
    Structure (stations per AP padded to a multiple of 4 to limit recompilations) and encoded observations of the
    current network.
    """

    width = int(4 * np.ceil(max(len(stas) for stas in scenario.associations.values()) / 4))
    st = create_structure(scenario, width)
    return {'scenario': scenario, 'st': st, 'contexts': pipe.contexts(key, scenario, st, n_contexts)}


def to_configs(st, choice, power, mcs):
    selected, tx_power, mcs = choice_to_arrays(st, *map(jnp.asarray, (choice, power, mcs)))
    return list(zip(np.asarray(selected_to_tx(st, selected)), np.asarray(tx_power), np.asarray(mcs)))


def select_refine(args, pipe, s_params, state, decisions, designated, scores, rng, refine):
    """
    Top-k covering the designated stations, then (if `refine`) surrogate-guided refinement with the designated links
    fixed. Returns the configurations (tx matrix, power, MCS).
    """

    st, designated = state['st'], np.asarray(designated)
    cover = (designated[..., 1, None] == st.link_sta).any(axis=-2)
    idxs = select_covering(scores, cover, args.top_k, 1.)
    decisions = [a[idxs] for a in decisions]

    if refine:
        fixed = designated_links(st, designated[idxs])
        *decisions, _ = pipe.refine(s_params, st, *decisions, fixed, rng, args.n_rounds)

    return to_configs(st, *decisions)


def regeneration_steps(traj, n_steps, interval_steps, enterprise):
    steps = {0} | (set(range(0, n_steps, interval_steps)) if interval_steps is not None else set())

    if enterprise:
        steps |= set(s for s in traj['change_steps'] if s < n_steps)

    return sorted(steps)


def run_zero_shot(args, base, traj, evaluate, enterprise, n_steps):
    """
    Zero-shot mode: the top-k of `n_samples` configurations (raw) and the refined top-k are generated on the
    regeneration grid of `interval_s` (plus topology changes in the enterprise scenario) and transmitted in turn.
    Longer intervals (`multiples` of `interval_s`) and no regeneration reuse the pools of their own regeneration steps.
    """

    pipe = Pipeline(args.run, args.surrogate, 6, args.chunk_size, refine_batch_size=args.refine_batch_size)
    key, rng = jax.random.PRNGKey(args.seed), np.random.default_rng(args.seed)
    regen = regeneration_steps(traj, n_steps, args.interval_steps, enterprise)
    pools, times = {'raw': [], 'ref': []}, []

    for i, start in enumerate(regen):
        t0 = time.time()
        key, ctx_key, designated_key, gen_key = jax.random.split(key, 4)
        state = network_state(pipe, state_scenario(base, traj, start, enterprise), ctx_key, 1)
        designated = round_robin_designated(
            designated_key, state['scenario'].associations, args.n_designated, args.n_samples
        )
        contexts = take_graphs(state['contexts'], np.zeros(args.n_samples, dtype=int))
        _, *decisions = pipe.generate(*jax.random.split(gen_key), pipe.params, state['st'], contexts, designated)
        scores = pipe.scores(pipe.s_params, state['st'], *decisions)

        for kind in ('raw',) if args.no_refine else ('raw', 'ref'):
            pools[kind].append(select_refine(args, pipe, pipe.s_params, state, decisions, designated, scores, rng,
                                             kind == 'ref'))

        times.append(time.time() - t0)
        print(f'regen {i + 1}/{len(regen)} step {start} time {times[-1]:.2f}s', flush=True)

    events = set(regeneration_steps(traj, n_steps, None, enterprise))
    variants = {'once': events}

    if args.interval_steps is not None:
        for k in args.multiples:
            variants[f'{args.interval_s * k:g}s'] = events | set(range(0, n_steps, args.interval_steps * k))

    out = {'regen_steps': regen, 'regen_time': times}
    index = {s: i for i, s in enumerate(regen)}

    for name, steps in variants.items():
        starts = np.asarray(sorted(steps))
        active = np.searchsorted(starts, np.arange(n_steps), side='right') - 1
        offset = np.arange(n_steps) - starts[active]

        for kind, pool in pools.items():
            if pool:
                configs = [pool[index[starts[a]]][o % len(pool[index[starts[a]]])] for a, o in zip(active, offset)]
                out[f'{kind}_{name}'], _ = evaluate(np.arange(n_steps), configs)
                print(f'{kind}_{name}: {out[f"{kind}_{name}"].mean():.1f}', flush=True)

    kind = 'raw' if args.no_refine else 'ref'
    out['data_rate'] = out[f'{kind}_{args.interval_s:g}s' if args.interval_steps is not None else f'{kind}_once']
    return out


def run_online(args, base, traj, evaluate, enterprise, n_steps):
    """
    Online mode: every RL step samples `n_samples` configurations (groups of `group_size` sharing an observation of the
    current state and designated stations) and transmits them one after another; their rates drive DiffusionNFT and
    the surrogate adaptation. The top-k of each step's samples (by the surrogate before the update) is also evaluated
    over the same transmission slots (counterfactual "deployed" stream), refined on deploy steps.
    """

    pipe = Pipeline(args.run, args.surrogate, 6, args.chunk_size, refine_batch_size=args.refine_batch_size)
    n_groups = args.n_samples // args.group_size
    key, rng = jax.random.PRNGKey(args.seed), np.random.default_rng(args.seed)
    optimizer, s_optimizer = create_optimizer(args.lr, grad_clip=1.), create_optimizer(args.surrogate_lr, grad_clip=1.)
    nft_step = create_nft_step(pipe.fm, optimizer, 1., 5., 1e-4)
    surrogate_step = create_surrogate_step(pipe.surrogate, s_optimizer, 1.)
    params = params_old = params_ref = pipe.params
    s_params = pipe.s_params
    opt_state, s_opt_state = optimizer.init(params), s_optimizer.init(s_params)

    n_rl = n_steps // args.n_samples
    explore, deployed_raw = np.zeros(n_rl * args.n_samples), np.zeros(n_rl * args.n_samples)
    deployed = np.full(n_rl * args.n_samples, np.nan)
    deploy_steps = set(range(0, n_rl, args.deploy_every)) | set(range(n_rl - args.final_steps, n_rl))
    log = []

    for i in range(n_rl):
        t0 = time.time()
        slots = np.arange(i * args.n_samples, (i + 1) * args.n_samples)
        key, ctx_key, designated_key, gen_key, perm_key = jax.random.split(key, 5)
        state = network_state(pipe, state_scenario(base, traj, slots[0], enterprise), ctx_key, n_groups)
        st = state['st']
        designated = sample_designated(designated_key, state['scenario'].associations, args.n_designated, n_groups)
        designated = jnp.repeat(designated, args.group_size, axis=0)
        contexts = take_graphs(state['contexts'], np.repeat(np.arange(n_groups), args.group_size))
        z, *decisions = pipe.generate(*jax.random.split(gen_key), params_old, st, contexts, designated)
        pre_scores = pipe.scores(s_params, st, *decisions)
        thr, rates = evaluate(slots, to_configs(st, *decisions))
        explore[slots] = thr

        for refine, stream in ((False, deployed_raw), (True, deployed)):
            if refine and i not in deploy_steps:
                continue

            pool = select_refine(args, pipe, s_params, state, decisions, designated, pre_scores, rng, refine)
            stream[slots], _ = evaluate(slots, [pool[j % len(pool)] for j in range(len(slots))])

        rewards = jnp.asarray(nft_rewards(group_advantages(thr, args.group_size), 5.))
        cond = jax.vmap(designated_edges)(contexts, designated)
        link_rates = np.stack(rates)[:, st.link_ap] * np.asarray(choice_to_arrays(st, *map(jnp.asarray, decisions))[0])
        groups = np.arange(len(slots)) // args.group_size
        losses, s_losses = [], []

        for idx in np.asarray(jax.random.permutation(perm_key, len(slots))).reshape(-1, args.minibatch_size):
            key, step_key = jax.random.split(key)
            batch = (take_graphs(contexts, idx), z[idx], rewards[idx], params_old, params_ref, cond[idx])
            params, opt_state, loss, _ = nft_step(params, opt_state, step_key, batch)
            losses.append(float(loss))

        for _ in range(4):
            key, perm_key = jax.random.split(key)

            for idx in np.asarray(jax.random.permutation(perm_key, len(slots))).reshape(-1, args.minibatch_size):
                batch = (st, *(jnp.asarray(a[idx]) for a in decisions), jnp.asarray(link_rates[idx]), groups[idx])
                s_params, s_opt_state, s_loss, _ = surrogate_step(s_params, s_opt_state, None, batch)
                s_losses.append(float(s_loss))

        params_old, _ = old_params_update(params, params_old, i, 1e-3, 0.5)
        log.append({
            'step': i, 'explore': float(thr.mean()), 'best': float(thr.reshape(n_groups, -1).max(axis=-1).mean()),
            'deployed': float(np.nanmean(deployed[slots])) if i in deploy_steps else None,
            'deployed_raw': float(deployed_raw[slots].mean()),
            'rank_corr': float(np.corrcoef(pre_scores.argsort().argsort(), thr.argsort().argsort())[0, 1]),
            'loss': float(np.mean(losses)), 's_loss': float(np.mean(s_losses)), 'time': time.time() - t0
        })
        print(json.dumps(log[-1]), flush=True)

    return {'data_rate': explore, 'deployed': deployed, 'deployed_raw': deployed_raw, 'log': log}


def run_hmab(args, base, traj, evaluate, enterprise, n_steps):
    agent, reward, thr = None, 0., np.zeros(n_steps)

    with jax.default_device(jax.devices('cpu')[0]):
        for step in trange(n_steps, desc='H-MAB'):
            if step in traj['assoc_at']:
                agent, reward = create_agent(traj['assoc_at'][step], args.seed), 0.

            tx, tx_power = agent.sample(reward)
            scenario = state_scenario(base, traj, step, enterprise)
            thr[step] = float(simulate_schedule(jnp.asarray(traj['eval_keys'][step]), scenario, tx, tx_power)[0])
            reward = thr[step] / base.normalize_reward

    return {'data_rate': thr}


def run_sr_groups(args, base, traj, evaluate, enterprise, n_steps):
    rng = np.random.default_rng(args.seed)
    configs = []

    for step in trange(n_steps, desc='SR groups'):
        if step % args.sr_update_steps == 0 or step in traj['assoc_at']:
            scenario = state_scenario(base, traj, step, enterprise)
            associations = scenario.associations
            ranked = form_groups(associations, rssi_matrix(scenario), args.gamma, args.max_size)[:args.top_k]
            n = traj['pos'][step].shape[0]

        tx = np.zeros((n, n), dtype=np.int32)

        for a in ranked[step % len(ranked)]:
            tx[a, associations[a][rng.integers(len(associations[a]))]] = 1

        configs.append((tx, np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)))

    return {'data_rate': evaluate(np.arange(n_steps), configs, oracle=True)[0]}


METHODS = {'zero_shot': run_zero_shot, 'online': run_online, 'hmab': run_hmab, 'sr_groups': run_sr_groups}


if __name__ == '__main__':
    args = ArgumentParser(description='Co-SR in mobility scenarios: zero-shot and online pipeline, H-MAB, SR groups')
    add_env_args(args)
    args.add_argument('--method', choices=list(METHODS), required=True)
    args.add_argument('--seed', type=int, default=1)
    args.add_argument('--interval_s', type=float, default=None, help='Regeneration interval (default: once)')
    args.add_argument('--multiples', type=int, nargs='+', default=[1], help='Zero-shot: multiples of interval_s')
    args.add_argument('--spawn_until_s', type=float, default=19.5)
    args.add_argument('--run', type=str, default='runs/flow_matching')
    args.add_argument('--surrogate', type=str, default='runs/surrogate')
    args.add_argument('--n_designated', type=int, default=4)
    args.add_argument('--n_samples', type=int, default=128)
    args.add_argument('--top_k', type=int, default=8)
    args.add_argument('--n_rounds', type=int, default=2)
    args.add_argument('--no_refine', default=False, action='store_true')
    args.add_argument('--refine_batch_size', type=int, default=256)
    args.add_argument('--chunk_size', type=int, default=64)
    args.add_argument('--group_size', type=int, default=16)
    args.add_argument('--minibatch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=3e-4)
    args.add_argument('--surrogate_lr', type=float, default=3e-4)
    args.add_argument('--deploy_every', type=int, default=5)
    args.add_argument('--final_steps', type=int, default=20)
    args.add_argument('--gamma', type=float, default=20.)
    args.add_argument('--max_size', type=int, default=3)
    args.add_argument('--sr_update_steps', type=int, default=1)
    args.add_argument('--traj_dir', type=str, default='results/dynamic/traj')
    args.add_argument('--output', type=str, required=True)
    args = args.parse_args()

    env, n_steps = create_env(args)
    enterprise = args.env == 'enterprise'
    base = copy.copy(env.scenario)
    args.interval_steps = None if args.interval_s is None else max(1, int(round(args.interval_s / args.sim_dt)))

    os.makedirs(path(args.traj_dir), exist_ok=True)
    traj_path = path(os.path.join(args.traj_dir, (
        f'{args.env}_walls{int(not args.no_walls)}_room{args.room_size}_v{args.mobility_speed_min}-'
        f'{args.mobility_speed_max}_seed{args.scenario_seed}_n{n_steps}.pkl'
    )))

    if os.path.exists(traj_path):
        with open(traj_path, 'rb') as f:
            traj = pickle.load(f)
    else:
        spawn_until = None if args.spawn_until_s is None else int(round(args.spawn_until_s / args.sim_dt))
        traj = build_trajectory(env, n_steps, args.scenario_seed, spawn_until, enterprise)

        with open(traj_path + '.tmp', 'wb') as f:
            pickle.dump(traj, f)

        os.replace(traj_path + '.tmp', traj_path)

    start = time.time()
    out = METHODS[args.method](args, base, traj, Evaluator(base, traj, enterprise), enterprise, n_steps)
    out['wall_time'] = time.time() - start

    arrays = {k: np.asarray(v) for k, v in out.items() if k.startswith(('data_rate', 'deployed', 'raw_', 'ref_'))}
    meta = {k: v for k, v in out.items() if k not in arrays} | {
        'args': vars(args), 'n_steps': n_steps, 'sim_dt': args.sim_dt, 'change_steps': traj['change_steps'],
        'n_stations': [sum(len(v) for v in traj['assoc_at'][s].values()) for s in traj['change_steps']]
    }
    os.makedirs(os.path.dirname(path(args.output)), exist_ok=True)
    np.savez_compressed(path(args.output), meta=json.dumps(meta, default=float), **arrays)
    print(f'saved {args.output} mean {arrays["data_rate"].mean():.1f} wall {out["wall_time"]:.0f}s', flush=True)
