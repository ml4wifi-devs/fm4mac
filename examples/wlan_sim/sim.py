import os
os.environ['JAX_ENABLE_X64'] = 'True'

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from examples.wlan_sim.constants import (
    ACK,
    ACK_DURATION,
    AM,
    CCA_PRIMARY,
    CCA_SECONDARY,
    CTS,
    CTS_DURATION,
    DATA,
    DIFS,
    IDLE,
    NOISE_DBM,
    N_CHANNELS,
    N_WIDTHS,
    OP,
    PIFS,
    RTS,
    RTS_DURATION,
    SCB,
    SIFS,
    SIFS_ACK,
    SIFS_CTS,
    SIFS_DATA,
    SLOT,
    TIMEOUT
)
from examples.wlan_sim.phy import dbm_to_mw, default_slot_nodes, link_tables, mw_to_dbm, node_positions, path_gain


INF = 2 ** 62


@dataclass(frozen=True)
class SimParams:
    sim_time: int = 10_000_000
    tx_power: float = 20.
    cca: float = -82.
    capture_threshold: float = 10.
    packet_length: int = 12000
    n_agg: int = 64
    cw_min: int = 0
    cw_max: int = 15
    cw_stage_max: int = 5
    cw_adaptation: bool = False
    frequency: float = 5.
    cca_mode: str = '11ax'
    rts_cts: bool = True
    tie_order: str = 'id'
    dtype: str = 'float64'


@jax.tree_util.register_dataclass
@dataclass
class Context:
    gain: jax.Array
    primary: jax.Array
    width: jax.Array
    policy: jax.Array
    node_primary: jax.Array
    sta_mask: jax.Array
    slot_node: jax.Array
    sta_assoc: jax.Array
    n_packets: jax.Array
    data_duration: jax.Array
    announced_duration: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class State:
    t: jax.Array
    step: jax.Array
    phase: jax.Array
    phase_end: jax.Array
    dest: jax.Array
    level: jax.Array
    rts_level: jax.Array
    cts_level: jax.Array
    idle_since: jax.Array
    ok: jax.Array
    failed: jax.Array
    aborted: jax.Array
    bo: jax.Array
    difs_end: jax.Array
    stage: jax.Array
    nav_ap: jax.Array
    nav_sta: jax.Array
    n_attempts: jax.Array
    n_success: jax.Array
    n_packets: jax.Array
    sta_packets: jax.Array
    n_rts_fail: jax.Array
    n_data_fail: jax.Array
    width_hist: jax.Array
    data_hist: jax.Array
    n_nav: jax.Array
    nav_time: jax.Array


def block(primary, level):
    lo = (primary >> level) << level
    return lo, lo + (1 << level) - 1


def in_block(primary, level):
    lo, hi = block(primary, level)
    ch = jnp.arange(N_CHANNELS)
    return (ch >= lo) & (ch <= hi)


def select_channels(power, primary, width, policy, key, params):
    """
    Channel bonding decision at the end of a backoff (AP) or at the start of an RTS (STA). `power` is the power
    sensed per 20 MHz channel in dBm. Returns whether a transmission is possible and the log2 width of the block.
    """

    blocks = jax.vmap(in_block, in_axes=(None, 0))(primary, jnp.arange(N_WIDTHS))

    if params.cca_mode == '11ax':
        fail = [
            (power[primary] > CCA_PRIMARY[0]) | (power[primary ^ 1] > CCA_SECONDARY[0]),
            (blocks[1] & (power > CCA_PRIMARY[1])).any()
            | ((blocks[2] & ~blocks[1]) & (power > CCA_SECONDARY[1])).any(),
            (blocks[2] & (power > CCA_PRIMARY[2])).any()
            | ((blocks[3] & ~blocks[2]) & (power > CCA_SECONDARY[2])).any()
        ]
        am_level = width
        for level in range(N_WIDTHS - 1, 0, -1):
            am_level = jnp.where((am_level == level) & fail[level - 1], level - 1, am_level)
        am_possible = True
        secondary_busy = blocks[width] & (jnp.arange(N_CHANNELS) != primary) & (power >= CCA_SECONDARY[0])
        scb_possible = (power[primary] < CCA_PRIMARY[0]) & ~secondary_busy.any()
        pu_level = jax.random.randint(key, (), 0, am_level + 1)
        pu_possible = True
    else:
        feasible = ~(blocks & (power > params.cca)).any(axis=-1) & (jnp.arange(N_WIDTHS) <= width)
        am_level = jnp.where(feasible, jnp.arange(N_WIDTHS), 0).max()
        am_possible = feasible[0]
        scb_possible = feasible[width]
        pu_level = jax.random.randint(key, (), 0, am_level + 1)
        pu_possible = feasible[0]

    op_possible = power[primary] < CCA_PRIMARY[0]
    policies = [policy == OP, policy == SCB, policy == AM]
    possible = jnp.select(policies, [op_possible, scb_possible, am_possible], pu_possible)
    level = jnp.select(policies, [0, width, am_level], pu_level)
    return possible, level


def draw_backoff(key, stage, params):
    scale = 2 ** stage if params.cw_adaptation else 1
    return jax.random.randint(key, stage.shape, params.cw_min * scale, params.cw_max * scale + 1)


def emitters(ctx, phase, dest, rts_level, params):
    n_aps = ctx.sta_mask.shape[0]
    from_sta = (phase == CTS) | (phase == ACK)
    sta_node = ctx.slot_node[jnp.arange(n_aps), dest]
    src = jnp.where(from_sta, sta_node, jnp.arange(n_aps))
    tx_power = (dbm_to_mw(params.tx_power) / jnp.where(from_sta, 1, 2 ** rts_level)).astype(params.dtype)
    rcv = jnp.where(from_sta, jnp.arange(n_aps), sta_node)
    active = (phase == RTS) | (phase == CTS) | (phase == DATA) | (phase == ACK)
    return src, tx_power, rcv, active


def sensed_power(ctx, primary, src, tx_power, level, active):
    received = tx_power[:, None] * ctx.gain[src] * active[:, None]
    channels = jax.vmap(in_block)(primary, level)
    return received, jnp.einsum('ir,ic->rc', received, channels)


def init_context(ap_pos, sta_pos, sta_mask, primary, width, policy, params, slot_node=None):
    n_aps, n_stas = sta_mask.shape
    slot_node = default_slot_nodes(n_aps, n_stas) if slot_node is None else slot_node
    gain = path_gain(node_positions(ap_pos, sta_pos), params.frequency)
    n_nodes = gain.shape[0]
    tx_power = jnp.full(n_aps, params.tx_power)
    _, _, n_packets, data_duration, announced = link_tables(
        gain, tx_power, n_aps, n_stas, params.n_agg, params.packet_length, slot_node
    )
    # STA nodes take the primary channel of the AP they are associated with
    active_node = jnp.where(sta_mask, slot_node, n_nodes)
    node_primary = jnp.zeros(n_nodes, dtype=primary.dtype).at[:n_aps].set(primary)
    node_primary = node_primary.at[active_node].set(jnp.broadcast_to(primary[:, None], sta_mask.shape), mode='drop')
    sta_assoc = jnp.zeros(n_nodes, dtype=bool).at[active_node].set(True, mode='drop')[n_aps:]
    return Context(
        gain.astype(params.dtype), primary, width, policy, node_primary, sta_mask, slot_node, sta_assoc, n_packets,
        data_duration, announced
    )


def init_state(key, ctx, params):
    n_aps, n_stas = ctx.sta_mask.shape
    zeros = jnp.zeros(n_aps, dtype=int)
    bo = draw_backoff(key, zeros, params)
    # An AP without associated STAs stays silent
    has_sta = ctx.sta_mask.any(axis=-1)
    state = State(
        t=jnp.array(0), step=jnp.array(0), phase=zeros + IDLE, phase_end=zeros + INF, dest=zeros, level=zeros,
        rts_level=zeros, cts_level=zeros, idle_since=jnp.zeros((ctx.gain.shape[0], N_CHANNELS), dtype=int),
        ok=jnp.zeros(n_aps, dtype=bool), failed=jnp.zeros(n_aps, dtype=bool),
        aborted=jnp.zeros(n_aps, dtype=bool), bo=bo,
        difs_end=jnp.where(has_sta, DIFS, INF), stage=zeros, nav_ap=zeros,
        nav_sta=jnp.zeros(ctx.sta_assoc.shape, dtype=int),
        n_attempts=zeros, n_success=zeros, n_packets=zeros, sta_packets=jnp.zeros((n_aps, n_stas), dtype=int),
        n_rts_fail=zeros, n_data_fail=zeros,
        width_hist=jnp.zeros((n_aps, N_WIDTHS), dtype=int), data_hist=jnp.zeros((n_aps, N_WIDTHS), dtype=int),
        n_nav=zeros, nav_time=zeros
    )
    state.t = next_event(state)
    return state


def next_event(state):
    idle = state.phase == IDLE
    counting = idle & (state.difs_end < INF)
    candidates = jnp.stack([
        jnp.where(~idle, state.phase_end, INF),
        jnp.where(counting, state.difs_end + state.bo * SLOT, INF),
        jnp.where(idle & (state.nav_ap > state.t), state.nav_ap, INF)
    ])
    return candidates.min()


def link_value(table, dest, rts_level, level):
    return table[jnp.arange(dest.shape[0]), dest, rts_level, level]


def step(key, ctx, s, params):
    n_aps = ctx.sta_mask.shape[0]
    aps = jnp.arange(n_aps)
    t = s.t
    key = jax.random.fold_in(key, s.step)
    bo_key, dest_key, ap_key, sta_key, tie_key = jax.random.split(key, 5)
    ap_keys, sta_keys = jax.random.split(ap_key, n_aps), jax.random.split(sta_key, n_aps)
    ce = 10. ** (params.capture_threshold / 10.)
    noise = dbm_to_mw(NOISE_DBM)
    cca = dbm_to_mw(params.cca)

    # Frames and phases ending at t
    src, tx_power, _, active = emitters(ctx, s.phase, s.dest, s.rts_level, params)
    ending = (s.phase != IDLE) & (s.phase_end == t)
    any_end = (ending & active).any()
    _, power = sensed_power(ctx, ctx.primary, src, tx_power, s.level, active & ~ending)
    power_dbm = mw_to_dbm(power + 1e-30)

    rts_end, cts_end, data_end, ack_end = (ending & (s.phase == p) for p in (RTS, CTS, DATA, ACK))
    gap_end = ending & ((s.phase == SIFS_CTS) | (s.phase == SIFS_DATA) | (s.phase == SIFS_ACK))
    rts_ok = rts_end & s.ok & (s.cts_level >= 0)
    rts_fail = rts_end & ~rts_ok
    data_fail = data_end & ~s.ok
    success = ack_end & s.ok
    restart = (cts_end & ~s.ok) | ack_end | (ending & (s.phase == TIMEOUT))
    failed = s.failed | rts_fail | (cts_end & ~s.ok) | data_fail | (ack_end & ~s.ok)

    level = jnp.where(rts_ok, s.cts_level, s.level)
    phase = jnp.select(
        [rts_ok, rts_fail | data_fail, cts_end & s.ok, data_end & s.ok, gap_end, restart],
        [SIFS_CTS, TIMEOUT, SIFS_DATA, SIFS_ACK, s.phase + 1, IDLE],
        s.phase
    )
    duration = jnp.select(
        [phase == CTS, phase == DATA, phase == ACK],
        [CTS_DURATION, link_value(ctx.data_duration, s.dest, s.rts_level, level), ACK_DURATION],
        SIFS
    )
    phase_end = jnp.where(ending, jnp.where(phase == IDLE, INF, t + duration), s.phase_end)

    # Backoff expiries at t
    counting = (s.phase == IDLE) & (s.difs_end < INF)
    expire = counting & (s.difs_end + s.bo * SLOT == t)
    rts_possible, rts_level = jax.vmap(select_channels, in_axes=(0, 0, 0, 0, 0, None))(
        power_dbm[:n_aps], ctx.primary, ctx.width, ctx.policy, ap_keys, params
    )
    start_rts = expire & rts_possible
    channel_fail = expire & ~rts_possible
    new_dest = jax.random.categorical(dest_key, jnp.where(ctx.sta_mask, 0., -jnp.inf), axis=-1)

    dest = jnp.where(start_rts, new_dest, s.dest)
    level = jnp.where(start_rts, rts_level, level)
    rts_level = jnp.where(start_rts, rts_level, s.rts_level)
    first_duration = RTS_DURATION if params.rts_cts else link_value(ctx.data_duration, dest, rts_level, rts_level)
    phase = jnp.where(start_rts, RTS if params.rts_cts else DATA, phase)
    phase_end = jnp.where(start_rts, t + first_duration, phase_end)
    failed = jnp.where(start_rts, False, failed)
    aborted = s.aborted & ~start_rts
    started = start_rts | gap_end

    # The CTS may only cover the secondary channels idle at the STA during the PIFS before the RTS started
    sta_node = ctx.slot_node[aps, dest]
    idle_since = s.idle_since[sta_node]
    free = ((idle_since >= 0) & (t - idle_since >= PIFS)) | (jnp.arange(N_CHANNELS) == ctx.primary[:, None])
    cts_possible, cts_level = jax.vmap(select_channels, in_axes=(0, 0, 0, 0, 0, None))(
        jnp.where(free, -jnp.inf, 0.), ctx.primary, rts_level, ctx.policy, sta_keys, params
    )
    cts_level = jnp.where(start_rts, jnp.where(cts_possible, cts_level, -1), s.cts_level)

    # Receptions of the frames on air after t
    src, tx_power, rcv, active = emitters(ctx, phase, dest, rts_level, params)
    received, power = sensed_power(ctx, ctx.primary, src, tx_power, level, active)
    lo, hi = block(ctx.primary, level)
    covers = (ctx.node_primary[None] >= lo[:, None]) & (ctx.node_primary[None] <= hi[:, None]) & active[:, None]
    interference = (received * covers).sum(axis=0)
    sinr = received / (interference[None] - received * covers + noise)
    decoded = covers & (sinr >= ce)

    # A node waiting for a CTS, DATA or ACK starts the reception ignoring interference; any transmission
    # starting later during the reception (also at the same instant, but processed later) triggers a SINR check
    rcv_decoded = decoded[aps, rcv]
    snr_ok = received[aps, rcv] >= ce * noise
    waiting = (phase == CTS) | (phase == ACK) | ((phase == DATA) & params.rts_cts)
    order = aps if params.tie_order == 'id' else jax.random.permutation(tie_key, n_aps)
    later_start = (started[None] & (order[None] > order[:, None])).any(axis=1)
    start_ok = jnp.where(waiting, snr_ok & (rcv_decoded | ~later_start), rcv_decoded)
    sta_free = s.nav_sta[sta_node - n_aps] <= t
    ok = jnp.where(started, start_ok & jnp.where(phase == RTS, sta_free, True), s.ok & (rcv_decoded | ~started.any()))

    # An AP whose CTS or ACK reception is interrupted restarts without changing its contention window
    interrupted = jnp.where(started, waiting & snr_ok, s.ok) & ~ok
    aborted = aborted | (interrupted & ((phase == CTS) | (phase == ACK)))

    # Third-party NAV
    exchange = (phase >= RTS) & (phase <= ACK)
    engaged = jnp.zeros(ctx.sta_assoc.shape, dtype=bool).at[sta_node - n_aps].max(exchange)
    eligible = jnp.concatenate([(phase == IDLE) & (s.nav_ap <= t), (s.nav_sta <= t) & ~engaged & ctx.sta_assoc])
    channel_busy = power > cca
    busy = channel_busy[jnp.arange(power.shape[0]), ctx.node_primary]
    rts_data = link_value(ctx.data_duration, dest, rts_level, rts_level)
    cts_data = link_value(ctx.announced_duration, dest, rts_level, level)
    cur_data = link_value(ctx.data_duration, dest, rts_level, level)
    nav_duration = jnp.select(
        [phase == RTS, phase == CTS, phase == DATA],
        [
            3 * SIFS + RTS_DURATION + CTS_DURATION + rts_data + ACK_DURATION,
            2 * SIFS + CTS_DURATION + cts_data + ACK_DURATION,
            SIFS + cur_data + ACK_DURATION
        ],
        ACK_DURATION
    )
    nav_set = started[:, None] & decoded & busy[None] & eligible[None]
    nav = jnp.where(
        nav_set.any(axis=0), t + (nav_set * nav_duration[:, None]).max(axis=0), jnp.concatenate([s.nav_ap, s.nav_sta])
    )
    nav_ap, nav_sta = nav[:n_aps], nav[n_aps:]
    ap_nav_set = nav_set.any(axis=0)[:n_aps]

    # Backoff
    stage = jnp.select(
        [restart & ~failed, restart & failed & ~s.aborted],
        [0, jnp.minimum(s.stage + 1, params.cw_stage_max)],
        s.stage
    )
    new_bo = draw_backoff(bo_key, stage, params)
    blocked = (nav_ap > t) | busy[:n_aps]
    still_counting = counting & ~expire
    freeze = still_counting & blocked
    bo = jnp.where(freeze, s.bo - jnp.maximum(t - s.difs_end, 0) // SLOT, s.bo)
    bo = jnp.select([restart, channel_fail], [new_bo + 1, new_bo], bo)
    difs_end = jnp.where(freeze | restart | channel_fail | start_rts, INF, s.difs_end)
    resume = (phase == IDLE) & (difs_end == INF) & ~blocked & (any_end | (s.nav_ap == t) | restart) & ~channel_fail
    resume = resume & ctx.sta_mask.any(axis=-1)
    difs_end = jnp.where(resume, t + DIFS, difs_end)

    delivered = success * link_value(ctx.n_packets, s.dest, s.rts_level, s.level)
    n_packets = s.n_packets + delivered
    width_hist = s.width_hist.at[aps, rts_level].add(start_rts)

    s = State(
        t=t, step=s.step + 1, phase=phase, phase_end=phase_end, dest=dest, level=level, rts_level=rts_level,
        cts_level=cts_level, idle_since=jnp.where(channel_busy, -1, jnp.where(s.idle_since < 0, t, s.idle_since)),
        ok=ok, failed=failed, aborted=aborted, bo=bo, difs_end=difs_end, stage=stage, nav_ap=nav_ap, nav_sta=nav_sta,
        n_attempts=s.n_attempts + start_rts, n_success=s.n_success + success, n_packets=n_packets,
        sta_packets=s.sta_packets.at[aps, s.dest].add(delivered),
        n_rts_fail=s.n_rts_fail + rts_fail + (cts_end & ~s.ok),
        n_data_fail=s.n_data_fail + data_fail + (ack_end & ~s.ok),
        width_hist=width_hist, data_hist=s.data_hist.at[aps, level].add(started & (phase == DATA)),
        n_nav=s.n_nav + ap_nav_set, nav_time=s.nav_time + jnp.where(ap_nav_set, nav_ap - t, 0)
    )
    s.t = next_event(s)
    return s


@partial(jax.jit, static_argnames=('params',))
def simulate(key, ap_pos, sta_pos, sta_mask, primary, width, policy, params=SimParams(), slot_node=None):
    """
    Event-driven simulation of downlink full-buffer CSMA/CA with RTS/CTS and dynamic channel bonding,
    following the Komondor v4.2.1 11ax model. Times in microseconds. Returns per-AP throughput in Mbps and stats.
    By default `sta_pos` has shape [n_aps, n_stas, 3]; with `slot_node` it lists the STA nodes and `slot_node`
    maps each (AP, slot) to a node index, so a STA can be a candidate of several APs (`sta_mask` selects the
    associated ones).
    """

    init_key, key = jax.random.split(key)
    ctx = init_context(ap_pos, sta_pos, sta_mask, primary, width, policy, params, slot_node)
    state = init_state(init_key, ctx, params)
    state = jax.lax.while_loop(lambda s: s.t < params.sim_time, lambda s: step(key, ctx, s, params), state)
    throughput = state.n_packets * params.packet_length / params.sim_time
    return throughput, state
