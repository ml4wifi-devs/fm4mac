import os
os.environ['JAX_ENABLE_X64'] = 'True'

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from examples.wlan_sim.constants import AM
from examples.wlan_sim.phy import path_loss
from examples.wlan_sim.sim import SimParams, simulate
from fm4mac.metrics import pf_utility


TX_POWER = 20.
CCA_THRESHOLD = -82.
MAX_CANDIDATES = 6
TRAIN_PARAMS = SimParams(sim_time=1_000_000, cw_adaptation=True, tie_order='random', dtype='float32')


@dataclass
class Deployment:
    ap_pos: np.ndarray
    sta_pos: np.ndarray
    primary: np.ndarray
    width: np.ndarray


def rssi_matrix(pos, ap_pos, tx_power=TX_POWER):
    """
    RSSI [dBm] of each AP at each position (shape [n_positions, n_aps]).
    """

    return tx_power - np.asarray(path_loss(np.linalg.norm(pos[:, None] - ap_pos[None], axis=-1)))


def conflict_graph(ap_pos):
    """
    APs that sense each other above the CCA threshold, and the AP-AP RSSI.
    """

    rssi = rssi_matrix(ap_pos, ap_pos)
    return (rssi > CCA_THRESHOLD) & ~np.eye(len(ap_pos), dtype=bool), rssi


def candidates(sta_pos, ap_pos, threshold=CCA_THRESHOLD, max_candidates=MAX_CANDIDATES):
    """
    Candidate APs of each STA: the strongest `max_candidates` APs above the threshold, as a boolean [n_stas, n_aps].
    """

    rssi = rssi_matrix(sta_pos, ap_pos)
    rank = np.argsort(np.argsort(-rssi, axis=1), axis=1)
    return (rssi >= threshold) & (rank < max_candidates)


def channel_plan(ap_pos, n_channels, rng):
    """
    DSATUR colouring of the AP conflict graph with `n_channels` non-overlapping 20 MHz channels (least-interfered
    channel when the colours run out).
    """

    conflict, rssi = conflict_graph(ap_pos)
    n_aps = len(ap_pos)
    primary = np.full(n_aps, -1)

    for _ in range(n_aps):
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


def random_deployment(
        rng, n_aps, stas_per_ap, ap_distance, n_channels, cluster_frac=0.5, cluster_size=8, cluster_radius=2.
):
    """
    APs uniform in a square of side sqrt(n_aps) * ap_distance with a coloured 20 MHz channel plan. STAs: a fraction
    in hotspots (`cluster_size` STAs uniform in a disc around a uniform centre), the rest uniform in the square.
    STAs without any candidate AP are redrawn.
    """

    side = np.sqrt(n_aps) * ap_distance
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

    for _ in range(100):
        uncovered = ~candidates(sta_pos, ap_pos).any(axis=1)

        if not uncovered.any():
            break

        sta_pos[uncovered] = draw(int(uncovered.sum()), 0)
    else:
        raise RuntimeError('STAs without coverage')

    primary = channel_plan(ap_pos, n_channels, rng)
    return Deployment(ap_pos, sta_pos, primary, np.zeros(n_aps, dtype=int))


TEST_DEPLOYMENTS = {f'random/{n}': (n, 4., 6., 3, 9000 + n) for n in (4, 6, 8, 10, 12, 16, 20, 24, 32, 48)}
TEST_DEPLOYMENTS |= {
    f'random/{n}/{k}': (n, 4., 6., 3, 9000 + n + 100 * k) for n in (16, 24, 32, 48) for k in range(1, 8)
}


def get_deployment(name):
    n_aps, stas_per_ap, ap_distance, n_channels, seed = TEST_DEPLOYMENTS[name]
    return random_deployment(np.random.default_rng(seed), n_aps, stas_per_ap, ap_distance, n_channels)


def draw_deployment(
        rng, n_aps_range=(2, 12), stas_per_ap_range=(2., 6.), ap_distance_range=(4., 8.), n_channels_range=(1, 4)
):
    """
    A deployment of the training distribution.
    """

    return random_deployment(
        rng, int(rng.integers(n_aps_range[0], n_aps_range[1] + 1)), float(rng.uniform(*stas_per_ap_range)),
        float(rng.uniform(*ap_distance_range)), int(rng.integers(n_channels_range[0], n_channels_range[1] + 1)),
        cluster_frac=float(rng.uniform(0., 0.8))
    )


@jax.tree_util.register_dataclass
@dataclass
class Network:
    """
    A deployment in the simulator layout: every AP has one slot per candidate STA (padded) pointing to the STA
    node, so an association is only a mask over the slots and a batch of associations shares the link tables.
    """

    ap_pos: jax.Array
    sta_pos: jax.Array
    slot_sta: jax.Array
    primary: jax.Array
    width: jax.Array
    candidates: jax.Array
    rssi: jax.Array


def make_network(deployment):
    cand = candidates(deployment.sta_pos, deployment.ap_pos)
    slot_sta = np.full((len(deployment.ap_pos), int(cand.sum(axis=0).max())), -1)

    for a in range(len(deployment.ap_pos)):
        stas = np.flatnonzero(cand[:, a])
        slot_sta[a, :len(stas)] = stas

    return Network(
        jnp.asarray(deployment.ap_pos), jnp.asarray(deployment.sta_pos), jnp.asarray(slot_sta),
        jnp.asarray(deployment.primary), jnp.asarray(deployment.width), jnp.asarray(cand),
        jnp.asarray(rssi_matrix(deployment.sta_pos, deployment.ap_pos))
    )


def slot_mask(net, assoc):
    return (net.slot_sta >= 0) & (assoc[jnp.maximum(net.slot_sta, 0)] == jnp.arange(len(net.ap_pos))[:, None])


def per_sta(net, values):
    """
    Scatters per-slot values of the associated STAs to a per-STA vector.
    """

    n_stas = net.candidates.shape[0]
    idx = jnp.where(net.slot_sta >= 0, net.slot_sta, n_stas)
    return jnp.zeros(n_stas + 1, dtype=values.dtype).at[idx].add(values)[:n_stas]


@partial(jax.jit, static_argnames=('params',))
def evaluate(keys, net, assoc, params=TRAIN_PARAMS):
    """
    Per-STA downlink throughput [Mbps] of a batch of associations (AP index of each STA, shape [B, n_stas]).
    """

    policy = jnp.full_like(net.primary, AM)
    slot_node = len(net.ap_pos) + jnp.maximum(net.slot_sta, 0)

    def single(key, assoc):
        mask = slot_mask(net, assoc)
        _, state = simulate(key, net.ap_pos, net.sta_pos, mask, net.primary, net.width, policy, params, slot_node)
        thr = state.sta_packets * params.packet_length / params.sim_time
        return per_sta(net, jnp.where(mask, thr, 0.))

    return jax.vmap(single)(keys, assoc)


def score(key, net, plans, n_seeds, params=TRAIN_PARAMS, chunk_size=1024):
    """
    Mean PF objective and mean per-STA throughput of each association (rows of `plans`) over `n_seeds` simulation
    seeds. All plans share the same seeds (common random numbers), which reduces the variance of comparisons.
    """

    plans = np.asarray(plans)
    reps = jnp.asarray(np.repeat(plans, n_seeds, axis=0))
    keys = jnp.tile(jax.random.split(key, n_seeds), (len(plans), 1))
    thr = np.concatenate([
        np.asarray(evaluate(keys[i:i + chunk_size], net, reps[i:i + chunk_size], params))
        for i in range(0, len(reps), chunk_size)
    ])
    thr = thr.reshape(len(plans), n_seeds, -1).mean(1)
    return np.asarray(pf_utility(thr)), thr


def option_table(net):
    """
    Candidate APs of each STA padded with -1, shape [n_stas, max_candidates]: the options of each STA in local
    search, refinement and bandits.
    """

    cand = np.asarray(net.candidates)
    table = np.full((len(cand), int(cand.sum(axis=1).max())), -1)

    for j, c in enumerate(cand):
        table[j, :c.sum()] = np.flatnonzero(c)

    return table


def random_assoc(rng, net):
    return np.array([rng.choice(np.flatnonzero(c)) for c in np.asarray(net.candidates)])
