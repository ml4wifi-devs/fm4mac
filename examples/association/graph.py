from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from examples.association.env import CCA_THRESHOLD, TX_POWER, candidates, rssi_matrix
from examples.wlan_sim.phy import path_loss
from fm4mac.graphs import JaxEnum, edge_attributes


RSSI_MEAN = -65.
RSSI_STD = 12.
STA_THROUGHPUT_SCALE = 50.
AP_THROUGHPUT_SCALE = 250.
LOAD_SCALE = 10.


class Associated(JaxEnum):
    FALSE = 0
    TRUE = 1
    NA = 2


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


class EdgeType(JaxEnum):
    SELF = 0
    CANDIDATE = 1
    AP_STA = 2
    AP_AP = 3
    NA = 4


@edge_attributes
class AssocEdge:
    assoc: Associated
    cochannel: float
    primary: Primary
    rssi: float
    throughput: float
    type: EdgeType


N_FEATURES = len(EdgeType) + len(Primary) + 6


@dataclass
class Topology:
    n_aps: int
    n_stas: int
    n_nodes: int
    n_candidates: int
    senders: np.ndarray
    receivers: np.ndarray
    rssi: np.ndarray
    type: np.ndarray
    cochannel: np.ndarray
    cand_ap: np.ndarray
    cand_sta: np.ndarray


def build_topology(deployment, threshold=CCA_THRESHOLD):
    """
    Nodes are the APs followed by the STAs. Edges, in this order: one self-loop per AP (channel and AP throughput),
    the candidate AP-STA links from AP to STA and then from STA to AP (association and STA throughput), other AP-STA
    pairs above the CCA threshold and AP-AP pairs above the CCA threshold (both directions).
    """

    ap_pos, sta_pos, primary = deployment.ap_pos, deployment.sta_pos, np.asarray(deployment.primary)
    n_aps, n_stas = len(ap_pos), len(sta_pos)
    cand = candidates(sta_pos, ap_pos)
    cand_sta, cand_ap = np.nonzero(cand)
    sta_rssi = rssi_matrix(sta_pos, ap_pos)
    other_sta, other_ap = np.nonzero(~cand & (sta_rssi > threshold))
    ap_rssi = TX_POWER - np.asarray(path_loss(np.linalg.norm(ap_pos[:, None] - ap_pos[None], axis=-1)))
    u, v = np.triu_indices(n_aps, k=1)
    keep = ap_rssi[u, v] > threshold
    u, v = u[keep], v[keep]

    aps = np.arange(n_aps)
    senders = np.concatenate([aps, cand_ap, n_aps + cand_sta, other_ap, n_aps + other_sta, u, v])
    receivers = np.concatenate([aps, n_aps + cand_sta, cand_ap, n_aps + other_sta, other_ap, v, u])
    rssi = np.concatenate([
        np.zeros(n_aps), *[sta_rssi[cand_sta, cand_ap]] * 2, *[sta_rssi[other_sta, other_ap]] * 2, *[ap_rssi[u, v]] * 2
    ])
    kind = np.concatenate([
        np.full(n_aps, EdgeType.SELF.value), np.full(2 * len(cand_ap), EdgeType.CANDIDATE.value),
        np.full(2 * len(other_ap), EdgeType.AP_STA.value), np.full(2 * len(u), EdgeType.AP_AP.value)
    ])
    cochannel = np.concatenate([
        np.zeros(n_aps + 2 * len(cand_ap) + 2 * len(other_ap)), *[(primary[u] == primary[v]).astype(float)] * 2
    ])

    return Topology(
        n_aps=n_aps, n_stas=n_stas, n_nodes=n_aps + n_stas, n_candidates=len(cand_ap), senders=senders,
        receivers=receivers, rssi=rssi, type=kind, cochannel=cochannel, cand_ap=cand_ap, cand_sta=cand_sta
    )


def to_graph(topology, primary, assoc, throughput=None, xp=jnp):
    """
    Autoencoder graph of an association (AP index of each STA) on a fixed topology, optionally with the observed
    per-STA throughput. With `xp=jnp` it is jit- and vmap-compatible, with `xp=np` it is cheap to build in data loaders.
    """

    n_aps, n_cand, n_edges = topology.n_aps, topology.n_candidates, len(topology.senders)
    one_hot = lambda x, n: xp.eye(n)[x]
    associated = (assoc[topology.cand_sta] == topology.cand_ap).astype(int)
    rest = n_edges - n_aps - 2 * n_cand

    if throughput is None:
        sta_thr, ap_thr = xp.zeros(n_cand), xp.zeros(n_aps)
    else:
        sta_thr = associated * throughput[topology.cand_sta] / STA_THROUGHPUT_SCALE
        ap_thr = xp.zeros(n_aps).at[assoc].add(throughput) if xp is jnp else np.bincount(assoc, throughput, n_aps)
        ap_thr = ap_thr / AP_THROUGHPUT_SCALE

    na = lambda n, enum: xp.full(n, len(enum) - 1)
    assoc = xp.concatenate([na(n_aps, Associated), associated, associated, na(rest, Associated)])
    edges = AssocEdge(
        assoc=one_hot(assoc, len(Associated)),
        cochannel=xp.asarray(topology.cochannel)[:, None],
        primary=one_hot(xp.concatenate([primary, na(n_edges - n_aps, Primary)]), len(Primary)),
        rssi=xp.where(topology.type == EdgeType.SELF.value, 0., (topology.rssi - RSSI_MEAN) / RSSI_STD)[:, None],
        throughput=xp.concatenate([ap_thr, sta_thr, sta_thr, xp.zeros(rest)])[:, None],
        type=one_hot(topology.type, len(EdgeType))
    )

    return jraph.GraphsTuple(
        nodes=xp.empty((topology.n_nodes, 0)), edges=edges,
        senders=xp.asarray(topology.senders), receivers=xp.asarray(topology.receivers), globals=None,
        n_node=xp.asarray([topology.n_nodes]), n_edge=xp.asarray([n_edges])
    )


def assoc_scores(assoc_logits, topology):
    """
    Association score (TRUE minus FALSE logit) of the candidate AP-to-STA edges, which follow the AP self-loops.
    """

    logits = assoc_logits[topology.n_aps:topology.n_aps + topology.n_candidates]
    return logits[..., Associated.TRUE.value] - logits[..., Associated.FALSE.value]


def sample_assoc(key, scores, topology):
    """
    One AP per STA, sampled from the softmax of the scores of its candidate edges (Gumbel-max).
    """

    cand_ap, cand_sta = jnp.asarray(topology.cand_ap), jnp.asarray(topology.cand_sta)
    scores = scores + jax.random.gumbel(key, scores.shape, dtype=scores.dtype)
    best = jax.ops.segment_max(scores, cand_sta, topology.n_stas)
    winner = scores >= best[cand_sta]
    return jnp.full(topology.n_stas, -1).at[cand_sta].max(jnp.where(winner, cand_ap, -1))


def plan_edges(topology, primary, assoc, xp=np):
    """
    Explicit surrogate features of an association: edge type, RSSI, channel on the AP self-loops, co-channel AP pairs,
    association flag of the candidate links, load (number of associated STAs) of the AP side of self-loops and
    AP-STA links, whether the AP of an AP-STA link is a co-channel interferer of the STA's serving AP, and the
    candidate flag.
    """

    n_aps = topology.n_aps
    is_self = topology.type == EdgeType.SELF.value
    is_ap_sta = (topology.type == EdgeType.CANDIDATE.value) | (topology.type == EdgeType.AP_STA.value)
    ap = xp.where(topology.senders < n_aps, topology.senders, topology.receivers)
    sta = xp.where(topology.senders < n_aps, topology.receivers, topology.senders) - n_aps
    sta = xp.where(is_ap_sta, sta, 0)

    load = xp.zeros(n_aps).at[assoc].add(1.) if xp is jnp else np.bincount(assoc, minlength=n_aps).astype(float)
    serving = assoc[sta]
    associated = is_ap_sta & (serving == ap)
    interferer = is_ap_sta & (serving != ap) & (primary[ap] == primary[serving])
    primary_hot = xp.where(
        is_self[:, None], xp.eye(len(Primary))[xp.where(is_self, primary[ap], 0)], xp.eye(len(Primary))[-1]
    )
    rssi = xp.where(is_self, 0., (topology.rssi - RSSI_MEAN) / RSSI_STD)

    return xp.concatenate([
        xp.eye(len(EdgeType))[topology.type], primary_hot, rssi[:, None], xp.asarray(topology.cochannel)[:, None],
        associated[:, None], interferer[:, None], (is_self | is_ap_sta)[:, None] * load[ap][:, None] / LOAD_SCALE,
        (topology.type == EdgeType.CANDIDATE.value)[:, None]
    ], axis=-1).astype(xp.float32)


def plan_graph(topology, primary, assoc, xp=np):
    """
    Explicit association graph scored by the surrogate.
    """

    return jraph.GraphsTuple(
        nodes=xp.empty((topology.n_nodes, 0), dtype=xp.float32), edges=plan_edges(topology, primary, assoc, xp),
        senders=xp.asarray(topology.senders), receivers=xp.asarray(topology.receivers), globals=None,
        n_node=xp.asarray([topology.n_nodes]), n_edge=xp.asarray([len(topology.senders)])
    )


def target_mask(topology, assoc, xp=np):
    """
    The AP-to-STA candidate edge of each STA's association, where the surrogate predicts the STA's log-throughput,
    over the first `n_aps + n_candidates` edges (AP self-loops and candidate AP-to-STA edges).
    """

    return xp.concatenate([xp.zeros(topology.n_aps, dtype=bool), assoc[topology.cand_sta] == topology.cand_ap])


def sta_targets(topology, thr, xp=np):
    """
    Log-throughput of each STA on its candidate edges (aligned with `target_mask`).
    """

    return xp.log(xp.maximum(thr, 1.))[..., xp.concatenate([xp.zeros(topology.n_aps, dtype=int), topology.cand_sta])]
