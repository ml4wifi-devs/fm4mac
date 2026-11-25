from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from examples.wlan_sim.phy import node_positions, path_loss
from fm4mac.graphs import JaxEnum, edge_attributes


CCA_THRESHOLD = -82.
RSSI_MEAN = -63.
RSSI_STD = 13.
THROUGHPUT_SCALE = 250.


class EdgeType(JaxEnum):
    SELF = 0
    ASSOC = 1
    AP_AP = 2
    AP_STA = 3
    STA_STA = 4
    NA = 5


class Primary(JaxEnum):
    CH_0 = 0
    CH_1 = 1
    CH_2 = 2
    CH_3 = 3
    CH_4 = 4
    CH_5 = 5
    CH_6 = 6
    CH_7 = 7
    NA = 8


class Width(JaxEnum):
    MHZ_20 = 0
    MHZ_40 = 1
    MHZ_80 = 2
    MHZ_160 = 3
    NA = 4


@edge_attributes
class ChannelEdge:
    primary: Primary
    rssi: float
    throughput: float
    type: EdgeType
    width: Width


@dataclass
class Topology:
    n_aps: int
    n_nodes: int
    wlan: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    rssi: np.ndarray
    type: np.ndarray


def build_topology(ap_pos, sta_pos, sta_mask, tx_power=20., threshold=CCA_THRESHOLD):
    """
    Nodes are the APs followed by their (valid) STAs. Edges: one self-loop per AP (carries the channel plan and
    the AP throughput), AP-STA association edges, and AP-AP, AP-STA and STA-STA edges between different WLANs
    whose 20 MHz RSSI exceeds the CCA threshold. All edges but the self-loops are added in both directions.
    """

    n_aps = len(ap_pos)
    sta_mask = np.asarray(sta_mask)
    wlan = np.concatenate([np.arange(n_aps), np.repeat(np.arange(n_aps), sta_mask.shape[1])[sta_mask.ravel()]])
    pos = np.asarray(node_positions(ap_pos, sta_pos))[np.concatenate([np.ones(n_aps, bool), sta_mask.ravel()])]
    is_ap = np.arange(len(pos)) < n_aps
    rssi = tx_power - np.asarray(path_loss(np.linalg.norm(pos[:, None] - pos[None], axis=-1)))

    u, v = np.triu_indices(len(pos), k=1)
    same = wlan[u] == wlan[v]
    kind = np.select(
        [same & (is_ap[u] | is_ap[v]), same, is_ap[u] & is_ap[v], is_ap[u] | is_ap[v]],
        [EdgeType.ASSOC.value, EdgeType.NA.value, EdgeType.AP_AP.value, EdgeType.AP_STA.value],
        EdgeType.STA_STA.value
    )
    keep = (kind == EdgeType.ASSOC.value) | ((kind != EdgeType.NA.value) & (rssi[u, v] > threshold))
    u, v, kind = u[keep], v[keep], kind[keep]

    aps = np.arange(n_aps)
    return Topology(
        n_aps=n_aps, n_nodes=len(pos), wlan=wlan,
        senders=np.concatenate([aps, u, v]), receivers=np.concatenate([aps, v, u]),
        rssi=np.concatenate([np.zeros(n_aps), rssi[u, v], rssi[u, v]]),
        type=np.concatenate([np.full(n_aps, EdgeType.SELF.value), kind, kind])
    )


def neighbourhoods(topology):
    """
    0/1 matrix of the WLANs connected by an edge (each WLAN is its own neighbour).
    """

    neighbours = np.eye(topology.n_aps)
    neighbours[topology.wlan[topology.senders], topology.wlan[topology.receivers]] = 1.
    return neighbours


def to_graph(topology, primary, width, throughput=None, xp=jnp):
    """
    Autoencoder graph of a channel plan (primary and log2 width per AP) with the observed AP throughput. With
    `xp=jnp` it is jit- and vmap-compatible, with `xp=np` it is cheap to build in data loaders.
    """

    n_aps, n_edges = topology.n_aps, len(topology.senders)
    pad = n_edges - n_aps
    one_hot = lambda x, n: xp.eye(n)[x]
    na = lambda x, enum: xp.concatenate([x, xp.full(pad, len(enum) - 1)])
    throughput = xp.zeros(n_aps) if throughput is None else throughput / THROUGHPUT_SCALE
    is_self = topology.type == EdgeType.SELF.value

    edges = ChannelEdge(
        primary=one_hot(na(primary, Primary), len(Primary)),
        rssi=xp.where(is_self, 0., (topology.rssi - RSSI_MEAN) / RSSI_STD)[:, None],
        throughput=xp.concatenate([throughput, xp.zeros(pad)])[:, None],
        type=one_hot(topology.type, len(EdgeType)),
        width=one_hot(na(width, Width), len(Width))
    )

    return jraph.GraphsTuple(
        nodes=xp.empty((topology.n_nodes, 0)), edges=edges,
        senders=xp.asarray(topology.senders), receivers=xp.asarray(topology.receivers), globals=None,
        n_node=xp.asarray([topology.n_nodes]), n_edge=xp.asarray([n_edges])
    )


def sample_plan(key, decoded, n_aps):
    """
    Channel plan sampled from the primary and width logits (NA class dropped) of the AP self-loops, which come first
    in the edge list.
    """

    primary_key, width_key = jax.random.split(key)
    primary = jax.random.categorical(primary_key, decoded.edges.primary[:n_aps, :-1])
    width = jax.random.categorical(width_key, decoded.edges.width[:n_aps, :-1])
    return primary, width


def plan_edges(topology, primary, width, xp=np):
    """
    Explicit surrogate features of channel plans (`primary` and `width` with optional leading batch axes): edge type,
    plan on the AP self-loops, RSSI, and for edges between different WLANs the channel overlap of the two operating
    blocks (overlap fraction seen from both ends and whether each block covers the other WLAN's primary channel).
    """

    a, b = topology.wlan[topology.senders], topology.wlan[topology.receivers]
    lo = (primary >> width) << width
    hi = lo + (1 << width) - 1
    lo_a, hi_a, lo_b, hi_b = lo[..., a], hi[..., a], lo[..., b], hi[..., b]
    primary_a, primary_b = primary[..., a], primary[..., b]
    is_self = topology.type == EdgeType.SELF.value
    overlap = xp.maximum(0, xp.minimum(hi_a, hi_b) - xp.maximum(lo_a, lo_b) + 1)
    shape = primary_a.shape

    static = np.concatenate([
        np.eye(len(EdgeType))[topology.type], np.where(is_self, 0., (topology.rssi - RSSI_MEAN) / RSSI_STD)[:, None]
    ], axis=-1)
    pairwise = (a != b)[:, None] * xp.stack([
        overlap / (hi_a - lo_a + 1), overlap / (hi_b - lo_b + 1),
        (lo_b <= primary_a) & (primary_a <= hi_b), (lo_a <= primary_b) & (primary_b <= hi_a)
    ], axis=-1)

    return xp.concatenate([
        xp.broadcast_to(static[:, :len(EdgeType)], shape + (len(EdgeType),)),
        xp.eye(len(Primary))[xp.where(is_self, primary_a, len(Primary) - 1)],
        xp.eye(len(Width))[xp.where(is_self, width[..., a], len(Width) - 1)],
        xp.broadcast_to(static[:, len(EdgeType):], shape + (1,)),
        pairwise
    ], axis=-1).astype(xp.float32)


def plan_graph(topology, primary, width, xp=np):
    """
    Explicit surrogate graph of channel plans; with leading batch axes in `primary` and `width` the plans share the
    structure (as expected by `fm4mac.surrogate.apply_batch`).
    """

    return jraph.GraphsTuple(
        nodes=xp.zeros(primary.shape[:-1] + (topology.n_nodes, 0), dtype=xp.float32),
        edges=plan_edges(topology, primary, width, xp),
        senders=xp.asarray(topology.senders), receivers=xp.asarray(topology.receivers), globals=None,
        n_node=xp.asarray([topology.n_nodes]), n_edge=xp.asarray([len(topology.senders)])
    )
