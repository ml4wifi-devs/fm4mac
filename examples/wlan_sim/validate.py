import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool

import jax
import jax.numpy as jnp
import numpy as np

from examples.wlan_sim.constants import AM, N_CHANNELS, N_WIDTHS, OP, SCB
from examples.wlan_sim.komondor import run
from examples.wlan_sim.phy import path_loss
from examples.wlan_sim.sim import SimParams, simulate


def channel_scenario(rng, key):
    """
    Random channel plan: APs uniform in a square of side sqrt(n_aps) * ap_distance, STAs uniform in an annulus of
    radii 1-5 m around their AP, random primary channels, widths and DCB policies (mixed or common).
    """

    n_aps = int(rng.choice([2, 4, 6, 8, 10]))
    n_stas = int(rng.choice([1, 2, 4]))
    ap_distance = float(rng.choice([5., 10., 15.]))
    ap_key, r_key, phi_key = jax.random.split(jax.random.fold_in(key, 0), 3)
    ap_pos = jax.random.uniform(ap_key, (n_aps, 2), maxval=jnp.sqrt(n_aps) * ap_distance)
    r = jnp.sqrt(jax.random.uniform(r_key, (n_aps, n_stas), minval=1., maxval=25.))
    phi = jax.random.uniform(phi_key, (n_aps, n_stas), maxval=2 * jnp.pi)
    sta_pos = (ap_pos[:, None] + jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi)], axis=-1)).reshape(-1, 2)
    primary_key, width_key = jax.random.split(jax.random.fold_in(key, 1))
    policy = rng.choice([OP, SCB, AM], size=n_aps) if rng.random() < 0.5 else np.full(n_aps, rng.choice([OP, SCB, AM]))
    return dict(
        ap_pos=np.pad(np.asarray(ap_pos), ((0, 0), (0, 1))), sta_pos=np.pad(np.asarray(sta_pos), ((0, 0), (0, 1))),
        serving=np.repeat(np.arange(n_aps), n_stas),
        primary=np.asarray(jax.random.randint(primary_key, (n_aps,), 0, N_CHANNELS)),
        width=np.asarray(jax.random.randint(width_key, (n_aps,), 0, N_WIDTHS)), policy=np.asarray(policy)
    )


def candidates(rssi, threshold=-82., max_candidates=6):
    rank = np.argsort(np.argsort(-rssi, axis=1), axis=1)
    return (rssi >= threshold) & (rank < max_candidates)


def rssi_matrix(pos, ap_pos):
    return 20. - np.asarray(path_loss(np.linalg.norm(pos[:, None] - ap_pos[None], axis=-1)))


def coloring(ap_pos, n_channels, rng):
    """
    DSATUR colouring of the AP conflict graph (APs sensing each other above -82 dBm) with `n_channels` 20 MHz
    channels; the least-interfered channel when the colours run out.
    """

    rssi = rssi_matrix(ap_pos, ap_pos)
    conflict = (rssi > -82.) & ~np.eye(len(ap_pos), dtype=bool)
    primary = np.full(len(ap_pos), -1)
    for _ in range(len(ap_pos)):
        uncolored = np.flatnonzero(primary < 0)
        saturation = [len(set(primary[conflict[i] & (primary >= 0)])) for i in uncolored]
        degree = conflict[uncolored].sum(1)
        i = uncolored[np.lexsort((rng.random(len(uncolored)), -degree, -np.array(saturation)))[0]]
        load = np.zeros(n_channels)
        for j in np.flatnonzero(conflict[i] & (primary >= 0)):
            load[primary[j]] += 10 ** (rssi[i, j] / 10)
        free = np.flatnonzero(load == 0)
        primary[i] = rng.choice(free) if len(free) else np.argmin(load)
    return primary


def association_scenario(rng, strongest, cluster_size=8, cluster_radius=2.):
    """
    Random association: 2-10 APs uniform in a square with a coloured 20 MHz channel plan, 2-6 STAs per AP, partly
    in hotspots, each associated with the strongest or a random candidate AP.
    """

    n_aps, stas_per_ap = int(rng.integers(2, 11)), float(rng.uniform(2., 6.))
    side = np.sqrt(n_aps) * float(rng.uniform(4., 8.))
    n_channels, cluster_frac = int(rng.integers(1, 5)), float(rng.uniform(0., 0.8))
    ap_pos = np.concatenate([rng.uniform(0, side, (n_aps, 2)), np.zeros((n_aps, 1))], axis=1)
    n_stas = int(round(n_aps * stas_per_ap))
    n_clustered = int(round(cluster_frac * n_stas))

    def draw(n_uniform, n_clustered):
        uniform = rng.uniform(0, side, (n_uniform, 2))
        centres = rng.uniform(0, side, (int(np.ceil(n_clustered / cluster_size)), 2))
        r = cluster_radius * np.sqrt(rng.uniform(size=n_clustered))
        phi = rng.uniform(0, 2 * np.pi, n_clustered)
        offsets = np.stack([r * np.cos(phi), r * np.sin(phi)], -1)
        clustered = np.repeat(centres, cluster_size, axis=0)[:n_clustered] + offsets
        return np.concatenate([np.concatenate([uniform, clustered]), np.zeros((n_uniform + n_clustered, 1))], axis=1)

    sta_pos = draw(n_stas - n_clustered, n_clustered)
    while not (covered := candidates(rssi_matrix(sta_pos, ap_pos)).any(axis=1)).all():
        sta_pos[~covered] = draw(int((~covered).sum()), 0)

    primary = coloring(ap_pos, n_channels, rng)
    rssi = rssi_matrix(sta_pos, ap_pos)
    cand = candidates(rssi)
    if strongest:
        serving = np.where(cand, rssi, -np.inf).argmax(1)
    else:
        serving = np.array([rng.choice(np.flatnonzero(c)) for c in cand])
    return dict(
        ap_pos=ap_pos, sta_pos=sta_pos, serving=serving, primary=primary, width=np.zeros(n_aps, dtype=int),
        policy=np.full(n_aps, AM), strongest=strongest
    )


def slots(serving, n_aps):
    """
    Simulator slot layout of the STAs served by each AP (padded); the identity layout for equal groups in order.
    """

    groups = [np.flatnonzero(serving == a) for a in range(n_aps)]
    slot_node = np.full((n_aps, max(len(g) for g in groups)), n_aps)
    for a, g in enumerate(groups):
        slot_node[a, :len(g)] = n_aps + g
    return slot_node, np.arange(slot_node.shape[1]) < np.array([len(g) for g in groups])[:, None]


def jax_throughput(keys, s, params):
    """
    Per-AP and per-STA throughput [Mbps] of a scenario for each key.
    """

    n_aps = len(s['ap_pos'])
    slot_node, mask = slots(s['serving'], n_aps)
    sim_fn = jax.vmap(simulate, in_axes=(0, None, None, None, None, None, None, None, None))
    ap_thr, state = sim_fn(
        keys, s['ap_pos'], s['sta_pos'], mask, s['primary'], s['width'], s['policy'], params, slot_node
    )
    sta_thr = np.zeros((len(keys), len(s['serving'])))
    sta_packets = np.asarray(state.sta_packets)[:, mask]
    sta_thr[:, slot_node[mask] - n_aps] = sta_packets * params.packet_length / params.sim_time
    return np.asarray(ap_thr), sta_thr


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('kind', choices=['channel', 'association'])
    args.add_argument('--n_scenarios', type=int, default=200)
    args.add_argument('--sim_time', type=float, default=10.)
    args.add_argument('--komondor_seeds', type=int, default=3)
    args.add_argument('--jax_seeds', type=int, default=10)
    args.add_argument('--n_workers', type=int, default=8)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--output', type=str, default=None)
    args = args.parse_args()

    params = SimParams(sim_time=int(args.sim_time * 1e6), cw_adaptation=True)
    rng = np.random.default_rng(args.seed)
    if args.kind == 'channel':
        keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_scenarios)
        scenarios = [channel_scenario(rng, key) for key in keys]
    else:
        scenarios = [association_scenario(rng, i % 2 == 0) for i in range(args.n_scenarios)]

    fields = ('ap_pos', 'sta_pos', 'serving', 'primary', 'width', 'policy')
    jobs = [(s, seed + 1) for s in scenarios for seed in range(args.komondor_seeds)]
    with ThreadPool(args.n_workers) as pool:
        komondor = pool.map(lambda job: run(*(job[0][f] for f in fields), params, job[1]), jobs)

    output = args.output or f'examples/wlan_sim/validation/{args.kind}.json'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    results = []
    for i, s in enumerate(scenarios):
        k_ap, k_sta = (np.stack(x) for x in zip(*komondor[i * args.komondor_seeds:(i + 1) * args.komondor_seeds]))
        j_ap, j_sta = jax_throughput(jax.random.split(jax.random.PRNGKey(i), args.jax_seeds), s, params)
        results.append(dict(
            {key: np.asarray(v).tolist() for key, v in s.items()},
            komondor_ap=k_ap.tolist(), komondor_sta=k_sta.tolist(), jax_ap=j_ap.tolist(), jax_sta=j_sta.tolist()
        ))
        print(f'{i:3d} aps={len(s["ap_pos"]):2d} stas={len(s["serving"]):3d} K={k_ap.mean(0).sum():7.1f} '
              f'J={j_ap.mean(0).sum():7.1f}', flush=True)

    with open(output, 'w') as f:
        json.dump(dict(kind=args.kind, args=vars(args), params=vars(params), results=results), f)
