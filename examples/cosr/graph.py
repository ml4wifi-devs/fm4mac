from dataclasses import dataclass
from itertools import combinations

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from mapc_sim.constants import DEFAULT_TX_POWER

from fm4mac.graphs import JaxEnum, bidirectional, edge_attributes


RSSI_MEAN = -51.6
RSSI_STD = 14.3
RSSI_THR = (-82. - RSSI_MEAN) / RSSI_STD
LINK_SCALE = 100.


class EdgeType(JaxEnum):
    AP = 0
    STA = 1
    NA = 2


class IsActive(JaxEnum):
    FALSE = 0
    TRUE = 1
    NA = 2


class IsSelected(JaxEnum):
    FALSE = 0
    TRUE = 1
    NA = 2


class McsValue(JaxEnum):
    MCS_0 = 0
    MCS_1 = 1
    MCS_2 = 2
    MCS_3 = 3
    MCS_4 = 4
    MCS_5 = 5
    MCS_6 = 6
    MCS_7 = 7
    MCS_8 = 8
    MCS_9 = 9
    MCS_10 = 10
    MCS_11 = 11
    MCS_12 = 12
    MCS_13 = 13
    NA = 14


class TxPowerValue(JaxEnum):
    LVL_1 = 0
    LVL_2 = 1
    LVL_3 = 2
    LVL_4 = 3
    NA = 4


N_POWER = len(TxPowerValue) - 1
N_MCS = len(McsValue) - 1
N_SETTINGS = N_POWER * N_MCS


@edge_attributes
class Connection:
    active: IsActive
    mcs: McsValue
    rssi: float
    selected: IsSelected
    success: float
    tx_power: TxPowerValue
    type: EdgeType


@jax.tree_util.register_dataclass
@dataclass
class Structure:
    """
    Static part of the Co-SR graph of a deployment. Edges: AP->STA links in association order, AP-AP pairs above
    the CCA threshold, then the same edges reversed. `ap_links[i, j]` is the link of the j-th station of the i-th AP
    (-1 = none), `edge_static` holds the edge type (AP, STA), the direction and the normalised RSSI. A pytree, so it can
    be an argument of jitted functions and topology changes do not recompile them.
    """

    ap_ids: np.ndarray
    ap_links: np.ndarray
    link_ap: np.ndarray
    link_sta: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    edge_static: np.ndarray
    is_ap: np.ndarray


def create_structure(scenario, width=None):
    """
    Structure of a mapc-sim scenario; `width` pads `ap_links` to a fixed number of stations per AP.
    """

    pos = np.asarray(scenario.pos)
    distance = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(axis=-1))
    rssi = (DEFAULT_TX_POWER - np.asarray(scenario.path_loss_fn(distance, scenario.walls)) - RSSI_MEAN) / RSSI_STD

    aps = [int(ap) for ap in scenario.associations]
    link_ap = np.asarray([ap for ap in aps for _ in scenario.associations[ap]], dtype=int)
    link_sta = np.asarray([int(sta) for ap in aps for sta in scenario.associations[ap]], dtype=int)
    pairs = np.asarray([(a, b) for a, b in combinations(aps, 2) if rssi[a, b] > RSSI_THR], dtype=int).reshape(-1, 2)
    senders, receivers = bidirectional(np.concatenate([link_ap, pairs[:, 0]]), np.concatenate([link_sta, pairs[:, 1]]))

    n_links, n_pairs = len(link_ap), len(pairs)
    edge_type = np.tile(np.repeat([EdgeType.STA.value, EdgeType.AP.value], [n_links, n_pairs]), 2)
    forward = np.repeat([1., 0.], n_links + n_pairs)
    edge_rssi = np.tile(np.concatenate([rssi[link_ap, link_sta], rssi[pairs[:, 0], pairs[:, 1]]]), 2)
    edge_static = np.concatenate([np.eye(2)[edge_type], forward[:, None], edge_rssi[:, None]], axis=-1)

    n_stas = [len(scenario.associations[ap]) for ap in aps]
    ap_links = np.full((len(aps), max(n_stas) if width is None else width), -1)

    for i, ap in enumerate(aps):
        ap_links[i, :n_stas[i]] = np.flatnonzero(link_ap == ap)

    return Structure(
        ap_ids=np.asarray(aps), ap_links=ap_links, link_ap=link_ap, link_sta=link_sta, senders=senders,
        receivers=receivers, edge_static=edge_static, is_ap=np.isin(np.arange(len(aps) + n_links), aps)
    )


def network_graph(st, selected, tx_power, mcs, success=None):
    """
    Network graph of the autoencoder: one schedule (selected links, per-node power and MCS indices, per-node success
    rate of the transmitted frames) on the structure of a deployment.
    """

    n_links, n_forward = st.link_ap.shape[0], st.senders.shape[0] // 2
    is_link = np.arange(n_forward) < n_links
    on = np.pad(np.asarray(selected, dtype=bool), (0, n_forward - n_links))
    ap = st.senders[:n_forward]
    success = np.zeros(st.is_ap.shape[0]) if success is None else np.asarray(success)
    one_hot = lambda idx, enum: np.tile(np.eye(len(enum))[idx], (2, 1))

    edges = Connection(
        active=one_hot(np.where(is_link, IsActive.TRUE.value, IsActive.NA.value), IsActive),
        mcs=one_hot(np.where(on, np.asarray(mcs)[ap], McsValue.NA.value), McsValue),
        rssi=st.edge_static[:, 3:],
        selected=one_hot(np.where(is_link, on, IsSelected.NA.value), IsSelected),
        success=np.tile(np.where(on, success[ap], 0.)[:, None], (2, 1)),
        tx_power=one_hot(np.where(on, np.asarray(tx_power)[ap], TxPowerValue.NA.value), TxPowerValue),
        type=one_hot(np.where(is_link, EdgeType.STA.value, EdgeType.AP.value), EdgeType)
    )
    n_node = st.is_ap.shape[0]
    return jraph.GraphsTuple(
        nodes=np.empty((n_node, 0)), edges=edges, senders=st.senders, receivers=st.receivers, globals=None,
        n_node=np.asarray([n_node]), n_edge=np.asarray([st.senders.shape[0]])
    )


def designated_edges(graph, designated):
    """
    Condition input of FM: 1 on the edges (both directions) of the designated (AP, STA) pairs `[m, 2]` (-1 = none).
    """

    senders, receivers = graph.senders[:, None], graph.receivers[:, None]
    aps, stas = designated[:, 0], designated[:, 1]
    pairs = ((senders == aps) & (receivers == stas)) | ((senders == stas) & (receivers == aps))
    return pairs.any(axis=-1, keepdims=True).astype(float)


def decide(key, edges, st, designated):
    """
    Per-AP decisions (link or -1 = off, power and MCS indices) of a decoded graph: each AP serves a random link decoded
    as selected, with power and MCS sampled from the link logits; designated (AP, STA) pairs `[m, 2]` (-1 = none) then
    override their APs.
    """

    tx_key, designate_key = jax.random.split(key)
    n_aps, n_links = st.ap_links.shape[0], st.link_ap.shape[0]
    selected = edges.selected[:n_links]
    valid = (jnp.argmax(edges.type[:n_links], axis=-1) == EdgeType.STA.value) & (selected[:, -1] <= 0)
    valid &= jnp.argmax(selected[:, :-1], axis=-1) == IsSelected.TRUE.value
    on = (st.ap_links >= 0) & valid[jnp.maximum(st.ap_links, 0)]
    n_valid = on.sum(axis=-1)

    keys = jax.random.split(tx_key, n_aps + 1)
    idx = jax.vmap(lambda k, n: jax.random.randint(k, (), 0, jnp.maximum(n, 1)))(keys[1:], n_valid)
    slot = jnp.argmax(on & (jnp.cumsum(on, axis=-1) - 1 == idx[:, None]), axis=-1)
    choice = jnp.where(n_valid > 0, st.ap_links[jnp.arange(n_aps), slot], -1)

    def split(key, _):
        key, power_key, mcs_key = jax.random.split(key, 3)
        return key, (power_key, mcs_key)

    _, (power_keys, mcs_keys) = jax.lax.scan(split, keys[0], length=n_aps)
    link = jnp.maximum(choice, 0)
    power = jnp.where(choice >= 0, jax.vmap(jax.random.categorical)(power_keys, edges.tx_power[link, :-1]), 0)
    mcs = jnp.where(choice >= 0, jax.vmap(jax.random.categorical)(mcs_keys, edges.mcs[link, :-1]), 0)

    def designate(decisions, inputs):
        key, (ap, sta) = inputs
        power_key, mcs_key = jax.random.split(key)
        link = jnp.argmax((st.link_ap == ap) & (st.link_sta == sta))
        i = jnp.argmax(st.ap_ids == ap)
        values = (link, jax.random.categorical(power_key, edges.tx_power[link, :-1]),
                  jax.random.categorical(mcs_key, edges.mcs[link, :-1]))
        return tuple(jnp.where(ap >= 0, d.at[i].set(v), d) for d, v in zip(decisions, values)), None

    keys = jax.random.split(designate_key, designated.shape[0])
    return jax.lax.scan(designate, (choice, power, mcs), (keys, designated))[0]


def choice_to_arrays(st, choice, power, mcs):
    """
    Per-AP decisions (any leading batch dimensions) to selected links and per-node power and MCS indices.
    """

    n_nodes, n_links = st.is_ap.shape[0], st.link_ap.shape[0]
    selected = jax.nn.one_hot(choice, n_links, dtype=jnp.int32).sum(axis=-2, dtype=jnp.int32)
    nodes = jnp.zeros(choice.shape[:-1] + (n_nodes,), dtype=jnp.int32)
    tx_power = nodes.at[..., st.ap_ids].set(jnp.where(choice >= 0, power, 0).astype(jnp.int32))
    mcs = nodes.at[..., st.ap_ids].set(jnp.where(choice >= 0, mcs, 0).astype(jnp.int32))
    return selected, tx_power, mcs


def selected_to_tx(st, selected):
    n_nodes = st.is_ap.shape[0]
    tx = jnp.zeros(selected.shape[:-1] + (n_nodes, n_nodes), dtype=jnp.int32)
    return tx.at[..., st.link_ap, st.link_sta].set(selected)


def encode_decisions(choice, power, mcs):
    """
    One integer per AP decision (0 = off), the option values of refinement.
    """

    return jnp.where(choice >= 0, (choice + 1) * N_SETTINGS + power * N_MCS + mcs, 0)


def decode_decisions(codes):
    return codes // N_SETTINGS - 1, codes % N_SETTINGS // N_MCS, codes % N_MCS


def explicit_graph(st, selected, tx_power, mcs):
    """
    Graph of the explicit surrogate for schedules (any leading batch dimensions): link edges carry activity, power and
    MCS one-hots, every edge carries the activity and power of its endpoints, nodes their role, activity, power and MCS.
    """

    f32 = jnp.float32
    n_nodes, n_links = st.is_ap.shape[0], st.link_ap.shape[0]
    senders, receivers = st.senders, st.receivers
    selected = selected.astype(f32)
    batch = selected.shape[:-1]
    active = jnp.minimum(selected @ jax.nn.one_hot(st.link_ap, n_nodes, dtype=f32), 1.)
    received = jnp.minimum(selected @ jax.nn.one_hot(st.link_sta, n_nodes, dtype=f32), 1.)
    power_hot = jnp.eye(N_POWER, dtype=f32)[tx_power] * active[..., None]
    mcs_hot = jnp.eye(N_MCS, dtype=f32)[mcs] * active[..., None]

    link = jnp.concatenate([
        jnp.ones(batch + (n_links, 1), dtype=f32), power_hot[..., st.link_ap, :], mcs_hot[..., st.link_ap, :]
    ], axis=-1) * selected[..., None]
    link = jnp.concatenate([link, jnp.zeros(batch + (senders.shape[0] // 2 - n_links, link.shape[-1]), f32)], axis=-2)
    ap_state = jnp.concatenate([active[..., None], power_hot], axis=-1)
    edge_static = jnp.broadcast_to(st.edge_static.astype(f32), batch + st.edge_static.shape)
    is_ap = jnp.broadcast_to(st.is_ap.astype(f32)[:, None], batch + (n_nodes, 1))

    edges = jnp.concatenate([
        edge_static, jnp.concatenate([link, link], axis=-2), ap_state[..., senders, :], ap_state[..., receivers, :]
    ], axis=-1)
    nodes = jnp.concatenate([is_ap, active[..., None], received[..., None], power_hot, mcs_hot], axis=-1)
    return jraph.GraphsTuple(
        nodes=nodes, edges=edges, senders=jnp.asarray(senders), receivers=jnp.asarray(receivers), globals=None,
        n_node=jnp.asarray([n_nodes]), n_edge=jnp.asarray([senders.shape[0]])
    )
