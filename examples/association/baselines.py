import os
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
import json
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from examples.association.env import (
    TRAIN_PARAMS,
    conflict_graph,
    evaluate,
    get_deployment,
    make_network,
    option_table,
    random_assoc,
    score
)
from examples.wlan_sim.constants import ACK_DURATION, CTS_DURATION, DIFS, RTS_DURATION, SIFS, SLOT
from examples.wlan_sim.phy import bits_per_symbol, data_duration, max_aggregated, select_mcs
from fm4mac.metrics import device_metrics, pf_utility
from fm4mac.search import create_coordinate_ascent


ROOT = os.path.dirname(os.path.abspath(__file__))
HEURISTICS = ['strongest', 'estimated_throughput', 'model_pf', 'mab_egreedy', 'mab_sticky']


def strongest_signal(net):
    """
    Default 802.11 association: each STA joins the AP with the strongest RSSI.
    """

    return np.where(np.asarray(net.candidates), np.asarray(net.rssi), -np.inf).argmax(axis=1)


class AirtimeModel:
    """
    Analytic model of downlink full-buffer 802.11 with equal transmission opportunities per STA: an AP with STAs K
    serves each STA in a fraction 1/|K| of its TXOPs, so STA j gets s_a * b_j / sum_{k in K} T_k (the 802.11
    performance anomaly), where b_j and T_j are the bits and duration of a TXOP to j and s_a is the airtime share of
    the AP among the co-channel APs with STAs that it senses. Interference and hidden terminals are ignored.
    """

    def __init__(self, net, params):
        mcs = np.asarray(select_mcs(np.asarray(net.rssi)))[..., 0]
        bps = np.asarray(bits_per_symbol(mcs, 0))
        n = np.asarray(max_aggregated(bps, params.n_agg, params.packet_length))
        backoff = SLOT * (params.cw_min + params.cw_max) / 2
        overhead = DIFS + backoff + RTS_DURATION + CTS_DURATION + ACK_DURATION + 3 * SIFS
        self.bits = n * params.packet_length
        self.duration = np.asarray(data_duration(n, bps, params.packet_length)) + overhead
        self.candidates = np.asarray(net.candidates)
        primary = np.asarray(net.primary)
        self.cochannel = conflict_graph(np.asarray(net.ap_pos))[0] & (primary[:, None] == primary[None])

    def throughput(self, assoc):
        """
        Per-STA throughput [Mbps] of an association.
        """

        stas = np.arange(len(assoc))
        load = np.bincount(assoc, weights=self.duration[stas, assoc], minlength=self.cochannel.shape[0])
        share = 1. / (1. + (self.cochannel & (load > 0)[None]).sum(axis=1))
        return share[assoc] * self.bits[stas, assoc] / load[assoc]

    def utility(self, assoc):
        return np.log(np.maximum(self.throughput(assoc), 1.)).sum()


def best_response(model, init, rng, value_fn, max_rounds=20):
    """
    STAs in random order move to the candidate AP that maximises `value_fn(model, assoc, sta)` until no STA moves.
    """

    assoc = np.array(init)

    for _ in range(max_rounds):
        changed = False

        for j in rng.permutation(len(assoc)):
            old, options = assoc[j], np.flatnonzero(model.candidates[j])
            values = []

            for a in options:
                assoc[j] = a
                values.append(value_fn(model, assoc, j))

            assoc[j] = options[int(np.argmax(values))]
            changed |= assoc[j] != old

        if not changed:
            break

    return assoc


def estimated_throughput(model, init, rng):
    """
    Selfish best response on the airtime model (802.11k/v-style estimated-throughput association).
    """

    return best_response(model, init, rng, lambda m, assoc, j: m.throughput(assoc)[j])


def model_pf(model, init, rng):
    """
    Centralised PF association on the airtime model (coordinate ascent of the sum of log throughputs).
    """

    return best_response(model, init, rng, lambda m, assoc, j: m.utility(assoc))


@jax.tree_util.register_dataclass
@dataclass
class AgentState:
    counts: jax.Array
    sums: jax.Array
    sticky: jax.Array
    arm: jax.Array


def choose(key, state, table, epsilon, sticky):
    """
    epsilon-greedy over the valid arms of each STA; with stickiness (Carrascosa and Bellalta, Computer
    Communications 2020) a STA keeps its AP while its sticky counter is positive.
    """

    explore_key, arm_key = jax.random.split(key)
    valid = table >= 0
    mean = jnp.where(valid, state.sums / jnp.maximum(state.counts, 1), -jnp.inf)
    # Ties (e.g. unexplored arms) keep the current AP
    greedy = jnp.argmax(mean + 1e-6 * jax.nn.one_hot(state.arm, table.shape[1]), axis=-1)
    random_arm = jax.random.categorical(arm_key, jnp.where(valid, 0., -jnp.inf))
    explore = jax.random.uniform(explore_key, greedy.shape) < epsilon
    arm = jnp.where(explore, random_arm, greedy)
    return jnp.where(state.sticky > 0, state.arm, arm) if sticky else arm


def bandits(key, net, init, n_rounds=240, epsilon=0.1, sticky=0, demand=10., n_runs=1, params=TRAIN_PARAMS):
    """
    Decentralised AP selection: every STA is a bandit over its candidate APs and observes its normalised
    throughput min(thr / demand, 1) after each association round (all STAs act simultaneously). A satisfied STA
    (reward 1) resets its sticky counter to `sticky`, an unsatisfied one decrements it. Returns the final
    associations of `n_runs` independent runs.
    """

    table = jnp.asarray(option_table(net))
    stas = jnp.arange(table.shape[0])
    zeros = jnp.zeros(table.shape)
    init_arm = jnp.argmax(table == jnp.asarray(init)[:, None], axis=1)
    state = AgentState(zeros, zeros, jnp.zeros(table.shape[0], dtype=int), init_arm)

    def step(states, key):
        act_key, sim_key = jax.random.split(key)
        arm = jax.vmap(choose, in_axes=(0, 0, None, None, None))(
            jax.random.split(act_key, n_runs), states, table, epsilon, sticky > 0
        )
        assoc = table[stas, arm]
        reward = jnp.minimum(evaluate(jax.random.split(sim_key, n_runs), net, assoc, params) / demand, 1.)
        one_hot = jax.nn.one_hot(arm, table.shape[1])
        states = AgentState(
            states.counts + one_hot, states.sums + one_hot * reward[..., None],
            jnp.where(reward >= 1., sticky, jnp.maximum(states.sticky - 1, 0)), arm
        )
        return states, assoc

    states = jax.tree.map(lambda x: jnp.repeat(x[None], n_runs, axis=0), state)
    _, assoc = jax.lax.scan(step, states, jax.random.split(key, n_rounds))
    return np.asarray(assoc[-1])


def create_local_search(n_seeds, params=TRAIN_PARAMS):
    """
    Centralised local search: STA-wise coordinate ascent over the candidate APs, every option scored by the mean PF
    of `n_seeds` simulations (the same seeds for all options). Call with `args=(net, key)`.
    """

    def score_fn(args, candidates):
        net, key = args
        n, n_options, n_stas = candidates.shape
        plans = jnp.repeat(candidates.reshape(-1, n_stas), n_seeds, axis=0)
        keys = jnp.tile(jax.random.split(key, n_seeds), (n * n_options, 1))
        thr = evaluate(keys, net, plans, params).reshape(n, n_options, n_seeds, n_stas).mean(2)
        return pf_utility(thr)

    return create_coordinate_ascent(score_fn)


def result_path(deployment, suffix):
    return os.path.join(ROOT, 'results', f'{deployment.replace("/", "_")}_{suffix}.json')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--deployment', type=str, default='random/8')
    args.add_argument('--n_rounds', type=int, default=3)
    args.add_argument('--ls_seeds', type=int, default=8)
    args.add_argument('--eval_seeds', type=int, default=16)
    args.add_argument('--mab_runs', type=int, default=4)
    args.add_argument('--demand', type=float, default=4.)
    args = args.parse_args()

    net = make_network(get_deployment(args.deployment))
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    model = AirtimeModel(net, TRAIN_PARAMS)
    ssf = strongest_signal(net)
    plans, times = {'random': np.stack([random_assoc(rng, net) for _ in range(16)]), 'strongest': ssf[None]}, {}

    for name, fn in [('estimated_throughput', estimated_throughput), ('model_pf', model_pf)]:
        start = time.time()
        plans[name] = fn(model, ssf, rng)[None]
        times[name] = time.time() - start

    for name, epsilon, sticky in [('mab_egreedy', 0.05, 0), ('mab_sticky', 0.1, 2)]:
        start = time.time()
        plans[name] = bandits(key, net, ssf, epsilon=epsilon, sticky=sticky, demand=args.demand, n_runs=args.mab_runs)
        times[name] = time.time() - start

    # From the best model-based heuristic by the simulator
    start = time.time()
    heuristics = np.concatenate([plans['strongest'], plans['estimated_throughput'], plans['model_pf']])
    init = heuristics[int(np.argmax(score(key, net, heuristics, args.ls_seeds)[0]))]
    table = option_table(net)
    plans['local_search'], _ = create_local_search(args.ls_seeds)(
        (net, key), init[None], np.maximum(table, 0), table >= 0, rng, args.n_rounds
    )
    times['local_search'] = time.time() - start

    eval_key = jax.random.PRNGKey(7)
    results = {name: device_metrics(score(eval_key, net, p, args.eval_seeds)[1]) for name, p in plans.items()}

    for name, m in results.items():
        print(f'{name:22s}', {k: round(v, 2) for k, v in m.items()}, round(times.get(name, 0.)), flush=True)

    os.makedirs(os.path.dirname(result_path(args.deployment, 'baselines')), exist_ok=True)

    with open(result_path(args.deployment, 'baselines'), 'w') as f:
        plans = {k: v.tolist() for k, v in plans.items()}
        json.dump(dict(results=results, times=times, plans=plans, args=vars(args)), f)
