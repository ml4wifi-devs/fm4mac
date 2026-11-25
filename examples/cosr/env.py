from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from mapc_research.envs.scenario_impl import StaticScenario, random_scenario, residential_scenario
from mapc_sim.sim import network_data_rate

from examples.cosr.graph import N_MCS, N_POWER, N_SETTINGS, choice_to_arrays, network_graph, selected_to_tx


data_rate_fn = jax.jit(network_data_rate, static_argnames=('channel_width', 'return_internals', 'path_loss_fn'))


@partial(jax.jit, static_argnames=('channel_width', 'path_loss_fn', 'oracle'))
def batch_data_rate_fn(keys, tx, tx_power, mcs, pos, sigma, walls, channel_width, path_loss_fn, oracle=False):
    """
    Aggregate and per-node data rates (Mb/s) of a batch of schedules; `pos` may be batched too. With `oracle` the
    simulator picks the best MCS.
    """

    def single(key, tx, tx_power, mcs, pos):
        thr, internals = network_data_rate(
            key, tx, pos, None if oracle else mcs, tx_power, sigma, walls, channel_width, return_internals=True,
            path_loss_fn=path_loss_fn
        )
        return thr, internals.average_data_rate / 1e6

    return jax.vmap(single, in_axes=(0, 0, 0, 0, 0 if pos.ndim == 3 else None))(keys, tx, tx_power, mcs, pos)


def create_simulator(scenario, st, chunk_size=2048):
    """
    Simulates per-AP decisions `[n, n_aps]` (one key per schedule split from `key`) and returns the aggregate data
    rates `[n]` and the data rates of the links `[n, n_links]` (Mb/s).
    """

    def simulate(key, choice, power, mcs):
        selected, tx_power, mcs = (np.asarray(a) for a in choice_to_arrays(st, *map(jnp.asarray, (choice, power, mcs))))
        tx = np.asarray(selected_to_tx(st, selected))
        keys = jax.random.split(key, len(tx))
        thr, rates = zip(*[
            batch_data_rate_fn(
                keys[i:i + chunk_size], tx[i:i + chunk_size],
                scenario.tx_power - scenario.tx_power_delta * tx_power[i:i + chunk_size], mcs[i:i + chunk_size],
                scenario.pos, scenario.sigma, scenario.walls, scenario.channel_width, scenario.path_loss_fn
            ) for i in range(0, len(tx), chunk_size)
        ])
        return np.concatenate(thr), np.concatenate(rates)[:, st.link_ap] * selected

    return simulate


@partial(jax.jit, static_argnames=('n_aps',))
def _random_tx(key, ap_links, n_aps):
    keys = jax.random.split(key, n_aps * 4)
    n_stas = (ap_links >= 0).sum(axis=-1)
    active = jax.vmap(lambda k: jax.random.uniform(k) < 1. / n_aps)(keys[0::4]) & (n_stas > 0)
    slot = jax.vmap(lambda k, n: jax.random.randint(k, (), 0, jnp.maximum(n, 1)))(keys[1::4], n_stas)
    power = jax.vmap(lambda k: jax.random.randint(k, (), 0, N_POWER))(keys[2::4])
    mcs = jax.vmap(lambda k: jax.random.randint(k, (), 0, N_MCS))(keys[3::4])
    choice = jnp.where(active, ap_links[jnp.arange(n_aps), jnp.minimum(slot, n_stas - 1)], -1)
    return choice, jnp.where(active, power, 0), jnp.where(active, mcs, 0)


def random_tx(key, st):
    """
    Random schedule of the observations: each AP serves a random station with probability 1 / n_aps.
    """

    return _random_tx(key, st.ap_links, st.ap_links.shape[0])


def simulate_schedule(key, scenario, tx, tx_power, mcs=None):
    """
    Aggregate data rate and simulator internals of one schedule (tx matrix, power indices, MCS or None for the best
    MCS).
    """

    return data_rate_fn(
        key, tx, scenario.pos, mcs, scenario.tx_power - scenario.tx_power_delta * tx_power, scenario.sigma,
        scenario.walls, scenario.channel_width, return_internals=True, path_loss_fn=scenario.path_loss_fn
    )


def success_rate(internals):
    return np.asarray(internals.frames_transmitted) / np.maximum(np.asarray(internals.ampdu_size), 1)


def observe(tx_key, sim_key, scenario, st):
    """
    Network graph of one observation: a random schedule and the success rates of its transmissions.
    """

    with jax.default_device(jax.devices('cpu')[0]):
        selected, tx_power, mcs = choice_to_arrays(st, *random_tx(tx_key, st))
        _, internals = simulate_schedule(sim_key, scenario, selected_to_tx(st, selected), tx_power, mcs)

    return network_graph(st, np.asarray(selected), np.asarray(tx_power), np.asarray(mcs), success_rate(internals))


def random_choice(rng, st, n):
    """
    Random per-AP decisions with a random fraction of active APs.
    """

    n_aps = st.ap_links.shape[0]
    n_options = (st.ap_links >= 0).sum(axis=-1)
    active = rng.random((n, n_aps)) < rng.uniform(0.1, 0.9, (n, 1))
    links = st.ap_links[np.arange(n_aps), rng.integers(0, 1 << 30, (n, n_aps)) % n_options]
    return np.where(active, links, -1), rng.integers(0, N_POWER, (n, n_aps)), rng.integers(0, N_MCS, (n, n_aps))


def interleaved_stations(key, associations):
    """
    (AP, STA) pairs visiting the APs in random order, one station of each AP per round.
    """

    ap_key, sta_key = jax.random.split(key)
    aps = np.asarray(jax.random.permutation(ap_key, np.asarray(list(associations))))
    shifts = np.asarray(jax.random.randint(sta_key, (len(aps),), 0, 1000))
    n_rounds = max(len(stas) for stas in associations.values())
    return np.asarray([
        [ap, associations[ap][(r + shift) % len(associations[ap])]]
        for r in range(n_rounds) for ap, shift in zip(aps, shifts) if r < len(associations[ap])
    ])


def sample_designated(key, associations, n_designated, n):
    """
    Designated stations of `n` schedules: `n_designated` (AP, STA) pairs of distinct APs, `[n, n_designated, 2]`.
    """

    return jnp.stack([interleaved_stations(k, associations)[:n_designated] for k in jax.random.split(key, n)])


def round_robin_designated(key, associations, n_designated, n):
    """
    Designated stations of `n` consecutive schedules that visit all stations in turn.
    """

    stations = interleaved_stations(key, associations)
    idxs = (np.arange(n)[:, None] * n_designated + np.arange(n_designated)) % len(stations)
    return jnp.asarray(stations[idxs])


def designated_links(st, designated):
    """
    Designated (AP, STA) pairs `[n, m, 2]` (-1 = none) to the designated link of every AP `[n, n_aps]` (-1 = free).
    """

    designated = np.asarray(designated)
    aps, stas = designated[..., :1], designated[..., 1:]
    links = ((st.link_ap == aps) & (st.link_sta == stas)).argmax(axis=-1)
    ap_idxs = (st.ap_ids == aps).argmax(axis=-1)
    rows = np.broadcast_to(np.arange(len(designated))[:, None], links.shape)
    valid = designated[..., 0] >= 0
    fixed = np.full((len(designated), len(st.ap_ids)), -1)
    fixed[rows[valid], ap_idxs[valid]] = links[valid]
    return fixed


def refinement_options(st, fixed):
    """
    Options of every AP in refinement, encoded as in `encode_decisions`: off, or one of its stations with any power and
    MCS. An AP with a designated link (`fixed` >= 0) only changes its power and MCS.
    """

    links = np.repeat(st.ap_links, N_SETTINGS, axis=1)
    settings = np.tile(np.arange(N_SETTINGS), st.ap_links.shape[1])
    options = np.concatenate([np.zeros((len(links), 1), dtype=int), (links + 1) * N_SETTINGS + settings], axis=1)
    links = np.concatenate([np.full((len(links), 1), -1), links], axis=1)
    valid = (links >= 0) | (np.arange(links.shape[1]) == 0)
    valid = np.where(fixed[..., None] >= 0, links == fixed[..., None], valid)
    return options, valid


def open_enterprise_scenario(seed, n_steps, x_apartments, y_apartments, n_sta_per_ap, size, channel_width):
    """
    Grid of rooms with an AP in the centre of each room, stations placed at random within the rooms, no walls.
    """

    key = jax.random.PRNGKey(seed)
    associations, pos = {}, []

    for x, y in product(range(x_apartments), range(y_apartments)):
        ap = len(pos)
        associations[ap] = list(range(ap + 1, ap + n_sta_per_ap + 1))
        pos.append([(x + 0.5) * size, (y + 0.5) * size])
        key, sta_key = jax.random.split(key)
        stas = jax.random.uniform(sta_key, (n_sta_per_ap, 2)) * size + jnp.array([x * size, y * size])
        pos.extend(stas.tolist())

    str_repr = f'open_enterprise_{seed}_{x_apartments}_{y_apartments}_{n_sta_per_ap}_{size}'
    return StaticScenario(jnp.array(pos), associations, n_steps, str_repr=str_repr, channel_width=channel_width)


TEST_SCENARIOS = {
    'residential': [
        residential_scenario(seed=100 + 3 * i, n_steps=n, x_apartments=x, y_apartments=y, n_sta_per_ap=4, size=10.,
                             channel_width=80)
        for i, (x, y, n) in enumerate([
            (2, 2, 1500), (2, 3, 3000), (3, 3, 20000), (3, 4, 200000), (4, 4, 2000), (4, 5, 2000), (5, 5, 2000),
            (5, 6, 2000)
        ])
    ],
    'enterprise': [
        open_enterprise_scenario(seed=101 + 3 * i, n_steps=n, x_apartments=x, y_apartments=y, n_sta_per_ap=4,
                                 size=20., channel_width=80)
        for i, (x, y, n) in enumerate([
            (2, 2, 1500), (2, 3, 3000), (3, 3, 20000), (3, 4, 200000), (4, 4, 2000), (4, 5, 20000), (5, 5, 200000),
            (5, 6, 200000)
        ])
    ],
    'random': [
        random_scenario(seed=100 + i, d_ap=75., d_sta=d_sta, n_ap=n_ap, n_sta_per_ap=n_sta, n_steps=n, channel_width=80)
        for i, (d_sta, n_ap, n_sta, n) in enumerate([
            (8., 2, 5, 1000), (5., 3, 3, 1000), (5., 3, 4, 1000), (5., 4, 3, 2000), (4., 4, 4, 1000), (4., 5, 3, 3000)
        ] * 3 + [
            (8., 2, 5, 1000), (5., 3, 3, 1000), (5., 3, 4, 1000), (5., 4, 3, 2000), (4., 4, 4, 2000), (4., 5, 3, 3000)
        ])
    ]
}
TEST_SCENARIOS['scaling'] = [TEST_SCENARIOS['residential'][i] for i in (0, 2, 4)]

# T-Optimal (mapc-optimal 0.3.0, tabu pricing) aggregate data rates of the residential scenarios (Mb/s)
T_OPTIMAL = {'residential': [1201.0, 1321.1, 2245.9, 3074.6, 3362.7, 3867.3, 5224.4, 5752.8]}


def get_scenario(name):
    split, idx = name.split('/')
    return TEST_SCENARIOS[split][int(idx)]
