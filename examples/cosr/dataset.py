import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORMS'] = 'cpu'

from argparse import ArgumentParser
from inspect import signature

import jax
import numpy as np
from mapc_optimal import OptimizationType, positions_to_path_loss
from mapc_research.envs.scenario_impl import (
    dense_point_scenario,
    flow_in_the_middle_scenario,
    hidden_station_scenario,
    indoor_small_bsss_scenario,
    random_scenario,
    residential_scenario,
    small_office_scenario,
    spatial_reuse_scenario,
    test_scenario,
    toy_scenario_1,
    toy_scenario_2
)
from mapc_sim.constants import DATA_RATES
from tqdm import tqdm

from examples.cosr.baselines import TX_POWER_LEVELS, create_agent, create_solver
from examples.cosr.env import random_tx, simulate_schedule, success_rate
from examples.cosr.graph import choice_to_arrays, create_structure, selected_to_tx
from examples.cosr.pipeline import path
from fm4mac.data import save_dataset


# scenario generators of the training networks and the ranges [a, b) of their parameters
SCENARIOS = [
    (toy_scenario_1, {'d': (10, 31)}),
    (toy_scenario_2, {'d_ap': (10, 31), 'd_sta': (1, 11)}),
    (small_office_scenario, {'d_ap': (10, 21), 'd_sta': (1, 11)}),
    (random_scenario, {
        'd_ap': (20, 101), 'n_ap': (2, 11), 'd_sta': (1, 9), 'n_sta_per_ap': (1, 6), 'randomize': (0, 1)
    }),
    (residential_scenario, {'x_apartments': (2, 6), 'y_apartments': (2, 3), 'n_sta_per_ap': (1, 5), 'size': (5, 21)}),
    (hidden_station_scenario, {'d': (21, 51)}),
    (flow_in_the_middle_scenario, {'d': (1, 31)}),
    (dense_point_scenario, {'n_ap': (2, 11), 'n_associations': (1, 6)}),
    (spatial_reuse_scenario, {'d_ap': (10, 21), 'd_sta': (1, 11)}),
    (test_scenario, {'scale': (10, 31)}),
    (indoor_small_bsss_scenario, {
        'grid_layers': (3, 4), 'n_sta_per_ap': (3, 11), 'frequency_reuse': (3, 4), 'bss_radius': (5, 21)
    })
]


def draw_scenarios(key, n):
    """
    Random training networks; generators are drawn with probability proportional to their number of parameters.
    """

    n_params = np.asarray([len(ranges) for _, ranges in SCENARIOS])
    key, choice_key = jax.random.split(key)
    idxs = jax.random.choice(choice_key, len(SCENARIOS), (n,), p=n_params / n_params.sum()).tolist()

    for i in idxs:
        scenario_fn, ranges = SCENARIOS[i]
        key, scenario_key = jax.random.split(key)
        *param_keys, seed_key = jax.random.split(scenario_key, len(ranges) + 1)
        params = {p: jax.random.randint(k, (), *r) for (p, r), k in zip(ranges.items(), param_keys)}
        if 'seed' in signature(scenario_fn).parameters:
            params['seed'] = jax.random.randint(seed_key, (), 0, 2 ** 30)

        yield scenario_fn(channel_width=80, **params).split_scenario()[0][0]


def schedule(key, scenario, st, tx, tx_power, mcs=None):
    """
    Simulated data rate and stored configuration of a schedule: selected links, per-node power and MCS indices and
    success rates (with `mcs` None the simulator picks the MCS).
    """

    thr, internals = simulate_schedule(key, scenario, tx, tx_power, mcs)
    configuration = (
        np.asarray(tx, dtype=np.int8)[st.link_ap, st.link_sta], np.asarray(tx_power, dtype=np.int8),
        np.asarray(internals.mcs if mcs is None else mcs, dtype=np.int8), success_rate(internals).astype(np.float32)
    )
    return float(thr), configuration


def random_configurations(key, scenario, st, n, args):
    for _ in range(n):
        key, tx_key, sim_key = jax.random.split(key, 3)
        selected, tx_power, mcs = choice_to_arrays(st, *random_tx(tx_key, st))
        yield schedule(sim_key, scenario, st, selected_to_tx(st, selected), tx_power, mcs)[1]


def h_mab_configurations(key, scenario, st, n, args):
    """
    Configurations of H-MAB after convergence (second half of `args.h_mab_steps` steps).
    """

    agent = create_agent(scenario.associations, 42)
    reward, configurations = 0., []

    for _ in range(args.h_mab_steps):
        key, sim_key = jax.random.split(key)
        tx, tx_power = agent.sample(reward)
        thr, configuration = schedule(sim_key, scenario, st, tx, tx_power)
        configurations.append(configuration)
        reward = thr / scenario.normalize_reward

    configurations = configurations[args.h_mab_steps // 2:]
    return [configurations[i] for i in jax.random.choice(key, len(configurations), (n,), replace=False)]


def optimal_configurations(key, scenario, st, n, args):
    """
    Configurations of the optimal schedule of mapc-optimal (T-Optimal or F-Optimal) drawn with their time shares.
    """

    opt_type = OptimizationType.SUM if args.generator == 't_optimal' else OptimizationType.MAX_MIN
    solver = create_solver(scenario.associations, opt_type, args.solver, args.threads)

    try:
        solution, _ = solver(positions_to_path_loss(scenario.pos, scenario.walls), scenario.associations)
    except Exception:
        return []

    confs = list(solution['links'])
    shares = np.asarray(list(solution['shares'].values()))
    n_nodes = st.is_ap.shape[0]
    configurations = []

    for _ in range(n):
        key, conf_key, sim_key = jax.random.split(key, 3)
        c = confs[jax.random.choice(conf_key, len(shares), p=shares).item()]
        tx = np.zeros((n_nodes, n_nodes), dtype=int)
        tx_power, mcs = np.zeros(n_nodes, dtype=int), np.zeros(n_nodes, dtype=int)

        for link in solution['links'][c]:
            ap, sta = (int(node.split('_')[1]) for node in link)
            tx[ap, sta] = 1
            tx_power[ap] = np.abs(TX_POWER_LEVELS - solution['tx_power'][c][link]).argmin()
            mcs[ap] = np.abs(DATA_RATES[80] - solution['link_rates'][c][link]).argmin()

        configurations.append(schedule(sim_key, scenario, st, tx, tx_power, mcs)[1])

    return configurations


GENERATORS = {
    'random': random_configurations, 'h_mab': h_mab_configurations, 't_optimal': optimal_configurations,
    'f_optimal': optimal_configurations
}


if __name__ == '__main__':
    args = ArgumentParser(description='Co-SR training data: random networks with random, H-MAB or optimal schedules')
    args.add_argument('--generator', choices=list(GENERATORS), required=True)
    args.add_argument('--n_networks', type=int, default=1000)
    args.add_argument('--n_configurations', type=int, default=30)
    args.add_argument('--h_mab_steps', type=int, default=2000)
    args.add_argument('--solver', type=str, choices=['cplex', 'cbc'], default='cplex')
    args.add_argument('--threads', type=int, default=None)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--output', type=str, required=True)
    args = args.parse_args()

    scenarios_key, configurations_key = jax.random.split(jax.random.PRNGKey(args.seed))
    dataset = []

    for scenario in tqdm(list(draw_scenarios(scenarios_key, args.n_networks)), desc='Networks'):
        configurations_key, key = jax.random.split(configurations_key)
        st = create_structure(scenario)
        configurations = list(GENERATORS[args.generator](key, scenario, st, args.n_configurations, args))
        dataset.append({'scenario': scenario, 'configurations': configurations})

    os.makedirs(os.path.dirname(path(args.output)), exist_ok=True)
    save_dataset(dataset, path(args.output))
