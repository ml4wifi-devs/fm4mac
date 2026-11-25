import jax.numpy as jnp

from examples.wlan_sim.constants import (
    CODING_RATES,
    HE_SU_PREAMBLE,
    MAC_HEADER,
    MAX_PPDU_DURATION,
    MCS_THRESHOLDS,
    MODULATION_BITS,
    MPDU_DELIMITER,
    N_SUBCARRIERS,
    N_WIDTHS,
    OFDM_SYMBOL,
    SERVICE_FIELD
)


def path_loss(distance, frequency=5.):
    d = jnp.maximum(distance, 1e-3)
    n_floors = d / 3.
    n_walls = d / 10.
    return (
        40.05 + 20. * jnp.log10(frequency / 2.4) + 20. * jnp.log10(jnp.minimum(d, 5.))
        + jnp.where(d >= 5., 35. * jnp.log10(d / 5.), 0.)
        + 18.3 * n_floors ** ((n_floors + 2.) / (n_floors + 1.) - 0.46) + 5. * n_walls
    )


def dbm_to_mw(x):
    return 10. ** (x / 10.)


def mw_to_dbm(x):
    return 10. * jnp.log10(x)


def node_positions(ap_pos, sta_pos):
    return jnp.concatenate([ap_pos, sta_pos.reshape(-1, sta_pos.shape[-1])])


def path_gain(pos, frequency=5.):
    distance = jnp.linalg.norm(pos[:, None] - pos[None], axis=-1)
    gain = dbm_to_mw(-path_loss(distance, frequency))
    return jnp.where(jnp.eye(pos.shape[0], dtype=bool), 0., gain)


def select_mcs(rssi):
    offset = 3. * jnp.arange(N_WIDTHS)
    return (rssi[..., None, None] >= MCS_THRESHOLDS + offset[:, None]).sum(-1)


def bits_per_symbol(mcs, width):
    return jnp.asarray(N_SUBCARRIERS)[width] * jnp.asarray(MODULATION_BITS)[mcs] * jnp.asarray(CODING_RATES)[mcs]


def max_aggregated(bps, n_agg, packet_length):
    max_symbols = (MAX_PPDU_DURATION - HE_SU_PREAMBLE) // OFDM_SYMBOL
    n = jnp.floor((max_symbols * bps - SERVICE_FIELD) / (MPDU_DELIMITER + MAC_HEADER + packet_length))
    return jnp.clip(n, 0, n_agg).astype(int)


def fit_aggregated(bps, max_duration, packet_length):
    """
    Largest number of aggregated packets (at least one) whose DATA lasts at most `max_duration`.
    """

    max_symbols = (max_duration - HE_SU_PREAMBLE) // OFDM_SYMBOL
    n = (max_symbols * jnp.floor(bps) - SERVICE_FIELD) // (MPDU_DELIMITER + MAC_HEADER + packet_length)
    return jnp.maximum(n, 1).astype(int)


def data_duration(n, bps, packet_length):
    bps = jnp.floor(bps)
    bits = jnp.where(
        n == 1,
        SERVICE_FIELD + MAC_HEADER + packet_length,
        SERVICE_FIELD + n * (MPDU_DELIMITER + MAC_HEADER + packet_length)
    )
    return (HE_SU_PREAMBLE + jnp.ceil(bits / bps) * OFDM_SYMBOL).astype(int)


def default_slot_nodes(n_aps, n_stas):
    return n_aps + jnp.arange(n_aps * n_stas).reshape(n_aps, n_stas)


def link_tables(gain, tx_power, n_aps, n_stas, n_agg, packet_length, slot_node=None):
    """
    Per (AP, STA, RTS width, DATA width) number of aggregated packets, DATA duration, and DATA duration announced
    by the CTS. MCS is selected once from the interference-free RSSI at full power, as in Komondor. DATA sent on a
    width narrowed by the CTS keeps the MCS chosen for the RTS width and drops aggregated packets until it fits in
    the DATA duration announced by the RTS; the CTS announces the DATA with all packets. `slot_node` maps the STA
    slots of each AP to node indices (default: the STAs of each AP follow the APs in order).
    """

    slot_node = default_slot_nodes(n_aps, n_stas) if slot_node is None else slot_node
    rssi = tx_power[:, None] + mw_to_dbm(gain[jnp.arange(n_aps)[:, None], slot_node])
    mcs = select_mcs(rssi)
    widths = jnp.arange(N_WIDTHS)
    n = max_aggregated(bits_per_symbol(mcs, widths), n_agg, packet_length)
    bps = bits_per_symbol(mcs[..., :, None], widths)
    announced = data_duration(n[..., None], bps, packet_length)
    limit = jnp.minimum(jnp.diagonal(announced, axis1=-2, axis2=-1), MAX_PPDU_DURATION)
    n_data = jnp.minimum(fit_aggregated(bps, limit[..., None], packet_length), n[..., None])
    return rssi, mcs, n_data, data_duration(n_data, bps, packet_length), announced
