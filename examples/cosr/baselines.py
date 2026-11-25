import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORMS'] = 'cpu'

import json
from argparse import ArgumentParser
from itertools import chain

import jax
import numpy as np
import pulp as plp
from mapc_mab import MapcAgentFactory
from mapc_optimal import OptimizationType, Solver, positions_to_path_loss
from mapc_sim.constants import DEFAULT_TX_POWER, NOISE_FLOOR, REFERENCE_DISTANCE
from reinforced_lib.agents.mab import UCB
from tqdm import trange

from examples.cosr.env import TEST_SCENARIOS, simulate_schedule
from examples.cosr.graph import N_POWER
from examples.cosr.pipeline import path


TX_POWER_LEVELS = DEFAULT_TX_POWER - 3. * np.arange(N_POWER)


def create_agent(associations, seed):
    """
    Hierarchical multi-armed bandit (H-MAB) Co-SR agent.
    """

    return MapcAgentFactory(
        associations={int(ap): [int(s) for s in stas] for ap, stas in associations.items()},
        agent_type=UCB,
        agent_params_lvl1={'c': 1.5, 'gamma': 0.5},
        agent_params_lvl2={'c': 0.5, 'gamma': 0.5},
        agent_params_lvl3={'c': 0.2, 'gamma': 0.8},
        hierarchical=True,
        seed=seed
    ).create_mapc_agent()


def run_h_mab(scenario, seed, n_steps):
    """
    H-MAB with oracle MCS; its data rate is the mean over the last 20% of the steps (at least 200).
    """

    agent = create_agent(scenario.associations, seed)
    key = jax.random.PRNGKey(seed)
    reward, rates = 0., []

    for _ in trange(n_steps, desc='H-MAB'):
        key, sim_key = jax.random.split(key)
        tx, tx_power = agent.sample(reward)
        rate, _ = simulate_schedule(sim_key, scenario, tx, tx_power)
        rates.append(float(rate))
        reward = rates[-1] / scenario.normalize_reward

    window = min(n_steps, max(200, int(0.2 * n_steps)))
    return {'data_rate': float(np.mean(rates[-window:])), 'data_rates': rates}


def create_solver(associations, opt_type, solver='cplex', threads=None):
    """
    Column-generation solver of mapc-optimal (T-Optimal: `OptimizationType.SUM`, F-Optimal: `MAX_MIN`).
    """

    lp_solver = (plp.CPLEX_CMD if solver == 'cplex' else plp.PULP_CBC_CMD)(msg=False, threads=threads)

    if not lp_solver.available():
        raise RuntimeError(f'Solver {solver} is not available (CPLEX must be installed and on PATH); use --solver cbc')

    return Solver(
        list(chain.from_iterable(associations.values())), list(associations), max_tx_power=TX_POWER_LEVELS[0],
        min_tx_power=TX_POWER_LEVELS[-1], opt_type=opt_type, solver=lp_solver, channel_width=80
    )


def run_solver(scenario, opt_type, solver='cplex', threads=None):
    splits = []

    for static, duration in scenario.split_scenario():
        _, data_rate = create_solver(static.associations, opt_type, solver, threads)(
            positions_to_path_loss(static.pos, static.walls), static.associations
        )
        splits.append({'duration': duration, 'data_rate': data_rate})

    return {'data_rate': sum(s['data_rate'] * s['duration'] for s in splits) / sum(s['duration'] for s in splits),
            'splits': splits}


def run_dcf(scenario, name, seed, sim_time, warmup, spatial_reuse, output_dir):
    """
    Event-driven DCF (or 802.11ax OBSS/PD spatial reuse) simulation of mapc-dcf; results are written by its logger.
    """

    import simpy
    from mapc_dcf.channel import TAU, Channel
    from mapc_dcf.logger import Logger
    from mapc_dcf.nodes import AccessPoint

    sim_time = sim_time or 10000 * TAU
    parts = scenario.split_scenario()

    for i, (static, _) in enumerate(parts):
        part = f'{name}_{chr(ord("a") + i)}' if len(parts) > 1 else name
        logger = Logger(sim_time, warmup, os.path.join(output_dir, part))
        key, channel_key = jax.random.split(jax.random.PRNGKey(seed))
        des_env = simpy.Environment()
        channel = Channel(channel_key, spatial_reuse, static.channel_width, static.pos, static.walls)

        for ap, clients in static.associations.items():
            key, ap_key = jax.random.split(key)
            node = AccessPoint(ap_key, ap, static.pos, static.tx_power[ap], np.array(clients), channel, des_env, logger)
            node.start_operation(1)

        des_env.run(until=logger.warmup_length + sim_time)
        logger.dump_acumulators(1)
        logger.shutdown({'name': part, 'simulation_length': sim_time, 'warmup_length': warmup, 'n_runs': 1})


NOISE_LIN = 10 ** (NOISE_FLOOR / 10)


def estimated_sinr(a, s, group, rssi):
    interference = NOISE_LIN + sum(10 ** (rssi[j, s] / 10) for j in group if j != a)
    return rssi[a, s] - 10 * np.log10(interference)


def rssi_matrix(scenario):
    pos, walls, tx_power = np.asarray(scenario.pos), np.asarray(scenario.walls), np.asarray(scenario.tx_power)
    distance = np.clip(np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1)), REFERENCE_DISTANCE, None)
    return tx_power[:, None] - np.asarray(scenario.path_loss_fn(distance, walls))


def group_min_sinr(group, associations, rssi):
    return min(estimated_sinr(a, s, group, rssi) for a in group for s in associations[a])


def form_groups(associations, rssi, gamma, max_size):
    """
    Greedy SR groups: each AP starts a group and adds the APs that interfere least with it as long as the worst
    estimated SINR in the group stays above `gamma`. Groups are ranked by size, then by the worst SINR.
    """

    aps = list(associations)
    groups = []

    for ref in aps:
        group = [ref]

        for j in sorted((j for j in aps if j != ref), key=lambda j: max(rssi[j, s] for s in associations[ref])):
            if len(group) >= max_size:
                break
            if group_min_sinr(group + [j], associations, rssi) >= gamma:
                group.append(j)

        if tuple(sorted(group)) not in groups:
            groups.append(tuple(sorted(group)))

    return sorted(groups, key=lambda g: (len(g), group_min_sinr(g, associations, rssi)), reverse=True)


def run_sr_groups(scenario, n_steps, seed, gamma, max_size, top_k):
    """
    SR groups with oracle MCS: in each step the `top_k` best groups transmit one after another, each AP to a random
    associated station.
    """

    n_nodes = np.asarray(scenario.pos).shape[0]
    associations = {int(k): [int(s) for s in v] for k, v in scenario.associations.items()}
    ranked = form_groups(associations, rssi_matrix(scenario), gamma, max_size)
    key = jax.random.PRNGKey(seed)
    rates = []

    for _ in range(n_steps):
        for group in [ranked[i % len(ranked)] for i in range(top_k)]:
            tx = np.zeros((n_nodes, n_nodes), dtype=int)

            for a in group:
                key, sta_key = jax.random.split(key)
                tx[a, associations[a][int(jax.random.choice(sta_key, len(associations[a])))]] = 1

            key, sim_key = jax.random.split(key)
            rates.append(float(simulate_schedule(sim_key, scenario, tx, np.zeros(n_nodes, dtype=int))[0]))

    return {'data_rate': float(np.mean(rates)), 'data_rates': rates}


if __name__ == '__main__':
    args = ArgumentParser(description='Co-SR baselines on the test scenarios')
    args.add_argument('--method', choices=['h_mab', 't_optimal', 'f_optimal', 'dcf', 'sr', 'sr_groups'], required=True)
    args.add_argument('--scenarios', type=str, choices=list(TEST_SCENARIOS), default='residential')
    args.add_argument('--idx', type=int, default=None)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--solver', type=str, choices=['cplex', 'cbc'], default='cplex')
    args.add_argument('--threads', type=int, default=None)
    args.add_argument('--dcf_warmup', type=float, default=0.5)
    args.add_argument('--dcf_sim_time', type=float, default=None, help='Simulated time after the warmup (s)')
    args.add_argument('--sr_steps', type=int, default=64)
    args.add_argument('--gamma', type=float, default=20.)
    args.add_argument('--max_size', type=int, default=3)
    args.add_argument('--top_k', type=int, default=8)
    args.add_argument('--output_dir', type=str, default='results/baselines')
    args = args.parse_args()

    output_dir = path(os.path.join(args.output_dir, args.method))
    os.makedirs(output_dir, exist_ok=True)
    idxs = range(len(TEST_SCENARIOS[args.scenarios])) if args.idx is None else [args.idx]

    for i in idxs:
        scenario, name = TEST_SCENARIOS[args.scenarios][i], f'{args.scenarios}_{i}'

        if args.method in ('dcf', 'sr'):
            run_dcf(scenario, name, args.seed, args.dcf_sim_time, args.dcf_warmup, args.method == 'sr', output_dir)
            continue
        elif args.method == 'h_mab':
            result = run_h_mab(scenario, args.seed, scenario.n_steps)
        elif args.method == 'sr_groups':
            result = run_sr_groups(scenario, args.sr_steps, args.seed, args.gamma, args.max_size, args.top_k)
        else:
            opt_type = OptimizationType.SUM if args.method == 't_optimal' else OptimizationType.MAX_MIN
            result = run_solver(scenario, opt_type, args.solver, args.threads)

        print(name, len(scenario.associations), round(result['data_rate'], 1), flush=True)

        with open(os.path.join(output_dir, f'{name}.json'), 'w') as f:
            json.dump({'args': vars(args), **result}, f)
