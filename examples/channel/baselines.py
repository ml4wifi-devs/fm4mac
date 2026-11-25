import os
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
import json
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from examples.channel.env import (
    N_ACTIONS,
    encode_action,
    evaluate,
    get_deployment,
    objective,
    random_plan,
    score
)
from examples.wlan_sim.constants import N_CHANNELS, N_WIDTHS
from examples.wlan_sim.phy import mw_to_dbm, node_positions, path_gain
from fm4mac.metrics import device_metrics
from fm4mac.search import create_coordinate_ascent


ROOT = os.path.dirname(os.path.abspath(__file__))


def conflict_graph(ap_pos, sta_pos, tx_power=20., cca=-82.):
    """
    APs i and j conflict if one senses the other above the CCA threshold at 20 MHz.
    """

    n_aps = len(ap_pos)
    gain = np.asarray(path_gain(node_positions(ap_pos, sta_pos)))[:n_aps, :n_aps]
    rssi = tx_power + np.asarray(mw_to_dbm(np.maximum(gain, 1e-30)))
    return (rssi > cca) & ~np.eye(n_aps, dtype=bool), rssi


def blocks_overlap(p1, w1, p2, w2):
    lo1, lo2 = (p1 >> w1) << w1, (p2 >> w2) << w2
    return (lo1 <= lo2 + (1 << w2) - 1) & (lo2 <= lo1 + (1 << w1) - 1)


def coloring_plan(ap_pos, sta_pos, strict=True, rng=None):
    """
    DSATUR colouring of the AP conflict graph with 20 MHz channels (least-interfered channel when the colours
    run out). With `strict`, greedy widening: each AP takes the widest aligned block that overlaps no conflicting AP,
    otherwise every AP gets 160 MHz and relies on always-max DCB to back off on busy secondary channels.
    """

    rng = rng or np.random.default_rng(0)
    conflict, rssi = conflict_graph(ap_pos, sta_pos)
    n_aps = len(ap_pos)
    primary = np.full(n_aps, -1)

    for _ in range(n_aps):
        uncolored = np.flatnonzero(primary < 0)
        saturation = [len(set(primary[conflict[i] & (primary >= 0)])) for i in uncolored]
        degree = conflict[uncolored].sum(1)
        i = uncolored[np.lexsort((rng.random(len(uncolored)), -degree, -np.array(saturation)))[0]]
        load = np.zeros(N_CHANNELS)

        for j in np.flatnonzero(conflict[i] & (primary >= 0)):
            load[primary[j]] += 10 ** (rssi[i, j] / 10)

        free = np.flatnonzero(load == 0)
        primary[i] = rng.choice(free) if len(free) else np.argmin(load)

    if not strict:
        return encode_action(primary, np.full(n_aps, N_WIDTHS - 1))

    width = np.zeros(n_aps, dtype=int)

    for i in np.argsort(conflict.sum(1)):
        for w in range(N_WIDTHS - 1, -1, -1):
            if not any(blocks_overlap(primary[i], w, primary[j], width[j]) for j in np.flatnonzero(conflict[i])):
                width[i] = w
                break

    return encode_action(primary, width)


def local_search(key, deployment, init, kind='pf', n_rounds=2, n_seeds=2, rng=None, visited=False):
    """
    Centralised coordinate ascent with the simulator: APs in random order try all actions with the others fixed and
    keep the best (mean objective over `n_seeds` simulations). `init` is `[n, n_aps]`; returns the final plans, and
    with `visited` also every evaluated (plans, values) pair.
    """

    n_aps = init.shape[-1]

    def score_fn(keys, candidates):
        plans = jnp.repeat(candidates.reshape(-1, n_aps), n_seeds, axis=0)
        thr = evaluate(keys, *deployment, plans).reshape(-1, n_seeds, n_aps).mean(1)
        return objective(thr, kind).reshape(candidates.shape[:2])

    keys = jax.random.split(key, init.shape[0] * N_ACTIONS * n_seeds)
    options = np.tile(np.arange(N_ACTIONS), (n_aps, 1))
    search = create_coordinate_ascent(score_fn)
    plans, _, *history = search(keys, init, options, True, rng or np.random.default_rng(0), n_rounds, visited)
    return (plans, *history) if visited else plans


@jax.tree_util.register_dataclass
@dataclass
class BanditState:
    counts: jax.Array
    sums: jax.Array
    t: jax.Array


def bandit_mean(state):
    return state.sums / jnp.maximum(state.counts, 1)


def egreedy(key, state, epsilon=1.):
    """
    Decaying epsilon-greedy, epsilon_t = epsilon / sqrt(t), as in Barrachina-Munoz et al. (IEEE Access 2021).
    """

    explore_key, arm_key = jax.random.split(key)
    n_agents, n_arms = state.counts.shape
    explore = jax.random.uniform(explore_key, (n_agents,)) < epsilon / jnp.sqrt(state.t)
    random_arm = jax.random.randint(arm_key, (n_agents,), 0, n_arms)
    return jnp.where(explore, random_arm, jnp.argmax(bandit_mean(state), axis=-1))


def thompson(key, state, prior_std=1., noise_std=0.25):
    """
    Gaussian Thompson sampling with a N(0.5, prior_std^2) prior and known reward noise.
    """

    precision = 1. / prior_std ** 2 + state.counts / noise_std ** 2
    posterior_mean = (0.5 / prior_std ** 2 + state.sums / noise_std ** 2) / precision
    sample = posterior_mean + jax.random.normal(key, posterior_mean.shape) / jnp.sqrt(precision)
    return jnp.argmax(sample, axis=-1)


BANDITS = dict(egreedy=egreedy, thompson=thompson)


def decentralized_mab(key, deployment, policy='egreedy', n_iterations=200):
    """
    Each AP runs its own bandit over the 32 (primary, max width) arms; all act simultaneously and observe their own
    throughput normalised by their throughput alone at 160 MHz (Barrachina-Munoz et al., IEEE Access 2021).
    Returns the final greedy plan (best estimated arm of each AP).
    """

    ap_pos, sta_pos, sta_mask = deployment
    n_aps = len(ap_pos)
    norm_key, key = jax.random.split(key)
    alone = jnp.array([
        float(evaluate(
            jax.random.split(norm_key, 2), ap_pos[i:i + 1], sta_pos[i:i + 1], sta_mask[i:i + 1],
            jnp.full((2, 1), N_ACTIONS - 1)
        ).mean())
        for i in range(n_aps)
    ])

    def step(state, key):
        act_key, sim_key = jax.random.split(key)
        actions = BANDITS[policy](act_key, state)
        thr = evaluate(sim_key[None], ap_pos, sta_pos, sta_mask, actions[None])[0]
        one_hot = jax.nn.one_hot(actions, N_ACTIONS)
        rewards = jnp.minimum(thr / alone, 1.)
        return BanditState(state.counts + one_hot, state.sums + one_hot * rewards[:, None], state.t + 1), None

    zeros = jnp.zeros((n_aps, N_ACTIONS))
    state, _ = jax.lax.scan(step, BanditState(zeros, zeros, jnp.array(1)), jax.random.split(key, n_iterations))
    return np.asarray(jnp.argmax(bandit_mean(state), axis=-1))


def evaluate_plans(deployment, plans, n_seeds=8):
    """
    Per-AP throughput metrics of plans with the evaluation seeds shared by all methods.
    """

    return device_metrics(score(jax.random.PRNGKey(7), deployment, np.atleast_2d(plans), n_seeds)[1])


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--deployment', type=str, default='random/8')
    args.add_argument('--n_rounds', type=int, default=2)
    args.add_argument('--eval_seeds', type=int, default=8)
    args.add_argument('--output_dir', type=str, default='results')
    args = args.parse_args()

    deployment = get_deployment(args.deployment)
    ap_pos, sta_pos, _ = deployment
    key = jax.random.PRNGKey(0)
    plans = {
        'random': np.stack([random_plan(k, len(ap_pos)) for k in jax.random.split(key, 16)]),
        'coloring': coloring_plan(ap_pos, sta_pos, strict=True)[None]
    }
    times = {}
    start = coloring_plan(ap_pos, sta_pos, strict=False)[None]

    for kind in ('sum', 'pf'):
        t = time.time()
        plans[f'local_search_{kind}'] = local_search(key, deployment, start, kind, args.n_rounds)
        times[f'local_search_{kind}'] = time.time() - t

    for policy in BANDITS:
        t = time.time()
        plans[f'mab_{policy}'] = decentralized_mab(key, deployment, policy)[None]
        times[f'mab_{policy}'] = time.time() - t

    results = {name: evaluate_plans(deployment, p, args.eval_seeds) for name, p in plans.items()}

    for name, m in results.items():
        print(f'{name:20s}', {k: round(v, 2) for k, v in m.items()}, round(times.get(name, 0.)))

    output_dir = os.path.join(ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f'{args.deployment.replace("/", "_")}_baselines.json'), 'w') as f:
        json.dump(dict(results=results, times=times, plans={k: v.tolist() for k, v in plans.items()}), f, indent=1)
