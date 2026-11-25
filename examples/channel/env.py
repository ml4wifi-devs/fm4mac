import os
os.environ['JAX_ENABLE_X64'] = 'True'

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from examples.wlan_sim.constants import AM, N_CHANNELS, N_WIDTHS
from examples.wlan_sim.sim import SimParams, simulate


N_ACTIONS = N_CHANNELS * N_WIDTHS
TRAIN_PARAMS = SimParams(sim_time=1_000_000, cw_adaptation=True, tie_order='random', dtype='float32')

TEST_DEPLOYMENTS = {f'random/{n}': (n, 2, 7., 9000 + n) for n in (6, 8, 10, 12, 16, 20, 24, 32, 48)}
TEST_DEPLOYMENTS |= {f'random/{n}/{k}': (n, 2, 7., 9000 + n + 100 * k) for n in (16, 24, 32, 48) for k in (1, 2, 3)}


def decode_action(action):
    return action // N_WIDTHS, action % N_WIDTHS


def encode_action(primary, width):
    return primary * N_WIDTHS + width


def random_deployment(key, n_aps, n_stas, ap_distance=10., sta_radius=(1., 5.), height=0.):
    """
    APs uniform in a square of side sqrt(n_aps) * ap_distance, STAs uniform in an annulus around their AP.
    """

    ap_key, r_key, phi_key = jax.random.split(key, 3)
    side = jnp.sqrt(n_aps) * ap_distance
    ap_pos = jax.random.uniform(ap_key, (n_aps, 2), maxval=side)
    r = jnp.sqrt(jax.random.uniform(r_key, (n_aps, n_stas), minval=sta_radius[0] ** 2, maxval=sta_radius[1] ** 2))
    phi = jax.random.uniform(phi_key, (n_aps, n_stas), maxval=2 * jnp.pi)
    sta_pos = ap_pos[:, None] + jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi)], axis=-1)
    ap_pos = jnp.concatenate([ap_pos, jnp.full((n_aps, 1), height)], axis=-1)
    sta_pos = jnp.concatenate([sta_pos, jnp.zeros((n_aps, n_stas, 1))], axis=-1)
    return ap_pos, sta_pos


def get_deployment(name):
    """
    AP positions, STA positions and STA mask of a test deployment.
    """

    n_aps, n_stas, ap_distance, seed = TEST_DEPLOYMENTS[name]
    ap_pos, sta_pos = random_deployment(jax.random.PRNGKey(seed), n_aps, n_stas, ap_distance)
    return np.asarray(ap_pos), np.asarray(sta_pos), np.ones((n_aps, n_stas), dtype=bool)


def random_plan(key, n_aps):
    return np.asarray(jax.random.randint(key, (n_aps,), 0, N_ACTIONS))


@partial(jax.jit, static_argnames=('params',))
def evaluate(keys, ap_pos, sta_pos, sta_mask, actions, params=TRAIN_PARAMS):
    """
    Per-AP throughput [Mbps] of a batch of channel plans (actions of shape [B, n_aps]) under always-max DCB.
    """

    def single(key, action):
        primary, width = decode_action(action)
        thr, _ = simulate(key, ap_pos, sta_pos, sta_mask, primary, width, jnp.full_like(primary, AM), params)
        return thr

    return jax.vmap(single)(keys, actions)


def objective(thr, kind='pf'):
    """
    Proportional fairness (sum of log per-AP throughput, below 1 Mbps counts as 1 Mbps) or aggregate throughput.
    """

    if kind == 'pf':
        return jnp.log(jnp.maximum(thr, 1.)).sum(-1)
    if kind == 'sum':
        return thr.sum(-1)
    raise ValueError(kind)


def score(key, deployment, plans, n_seeds, kind='pf', chunk_size=4096):
    """
    Objective and per-AP throughput of each plan (rows of `plans`), averaged over `n_seeds` simulation seeds.
    """

    plans = np.repeat(np.asarray(plans), n_seeds, axis=0)
    keys = jax.random.split(key, len(plans))
    thr = np.concatenate([
        np.asarray(evaluate(keys[i:i + chunk_size], *deployment, plans[i:i + chunk_size]))
        for i in range(0, len(plans), chunk_size)
    ])
    thr = thr.reshape(-1, n_seeds, thr.shape[-1]).mean(1)
    return np.asarray(objective(thr, kind)), thr
