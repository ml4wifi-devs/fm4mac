import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from examples.cosr.env import observe, refinement_options
from examples.cosr.graph import (
    LINK_SCALE,
    Connection,
    choice_to_arrays,
    decide,
    decode_decisions,
    designated_edges,
    encode_decisions,
    explicit_graph
)
from fm4mac.autoencoder import Autoencoder
from fm4mac.checkpoints import load_config, restore_params
from fm4mac.flow import FlowMatching, add_condition_input, latent_graph
from fm4mac.graphs import stack_graphs
from fm4mac.rl import create_nft_loss
from fm4mac.search import create_coordinate_ascent
from fm4mac.surrogate import Surrogate, surrogate_loss, utility
from fm4mac.training import create_step_fn


ROOT = os.path.dirname(os.path.abspath(__file__))


def path(p):
    """
    Paths relative to the example directory.
    """

    return os.path.join(ROOT, p)


def load_autoencoder(ae_path):
    cfg = load_config(path(ae_path))
    model = Autoencoder(Connection, **cfg.model)
    params = restore_params(path(ae_path))
    encode_fn, decode_fn = (jax.jit(partial(model.apply, params, method=m)) for m in ('encode', 'decode'))
    return encode_fn, decode_fn


def surrogate_config(surrogate_path):
    """
    Model config and checkpoint item of the explicit surrogate of a surrogate run or of the online-adapted surrogate of
    an RL run.
    """

    cfg = load_config(path(surrogate_path))
    return (cfg.surrogate.model, 'surrogate_params') if 'surrogate' in cfg else (cfg.model, 'params')


def load_surrogate(surrogate_path, dtype='float32'):
    model_cfg, item = surrogate_config(surrogate_path)
    return Surrogate(**{**model_cfg, 'dtype': dtype}), restore_params(path(surrogate_path), item)


def default_surrogate(checkpoint):
    """
    The online-adapted surrogate of an RL run, otherwise the pretrained one.
    """

    return checkpoint if 'surrogate' in load_config(path(checkpoint)) else 'runs/surrogate'


def surrogate_scores(model, params, st, selected, tx_power, mcs, batch_size):
    """
    Predicted aggregate data rate (Mb/s) of schedules (any leading batch dimensions): the sum of the predicted data
    rates of their links.
    """

    def single(args):
        mean, _ = model.apply(params, explicit_graph(st, *args), training=False)
        return utility(mean, args[0].astype(mean.dtype))

    flat = [a.reshape((-1,) + a.shape[-1:]) for a in (selected, tx_power, mcs)]
    return jax.lax.map(single, flat, batch_size=batch_size).reshape(selected.shape[:-1]) * LINK_SCALE


def create_refiner(model, batch_size):
    """
    Refinement: coordinate ascent over the APs of many schedules at once with the surrogate as the scorer. Each AP is
    switched off or serves one of its stations with any power and MCS; APs with a designated link (`fixed` >= 0) keep
    it. Returns the refined decisions and their predicted data rates.
    """

    search = create_coordinate_ascent(lambda args, codes: surrogate_scores(
        model, args[0], args[1], *choice_to_arrays(args[1], *decode_decisions(codes)), batch_size
    ))

    def refine(s_params, st, choice, power, mcs, fixed, rng, n_rounds=2):
        options, valid = refinement_options(st, np.asarray(fixed))
        codes = encode_decisions(*map(jnp.asarray, (choice, power, mcs)))
        codes, scores = search((s_params, st), codes, options, valid, rng, n_rounds)
        return *(np.asarray(a) for a in decode_decisions(codes)), scores

    return refine


def create_surrogate_step(model, optimizer, rank_weight):
    """
    Online surrogate update on simulated schedules: batch = (structure, choice, power, MCS, link rates, groups).
    """

    def loss_fn(params, key, batch):
        st, choice, power, mcs, link_rates, groups = batch
        selected, tx_power, mcs = choice_to_arrays(st, choice, power, mcs)
        loss, (nll, rank, _, _) = surrogate_loss(
            model, params, explicit_graph(st, selected, tx_power, mcs), (link_rates / LINK_SCALE).astype(jnp.float32),
            selected.astype(jnp.float32), groups, rank_weight
        )
        return loss, {'nll': nll, 'rank': rank}

    return create_step_fn(loss_fn, optimizer)


def create_nft_step(model, optimizer, beta, adv_clip, kl_coef):
    """
    DiffusionNFT update of FM: batch = (contexts, latents, rewards, old params, reference params, conditions).
    """

    nft_loss = create_nft_loss(model, beta, adv_clip, kl_coef)

    def loss_fn(params, key, batch):
        loss, kl = nft_loss(params, key, *batch)
        return loss, {'kl': kl}

    return create_step_fn(loss_fn, optimizer)


class Pipeline:
    """
    Co-SR inference: encode one observation of the network (context), sample candidate latents with FM conditioned on
    designated (AP, STA) pairs, decode them into per-AP decisions (station or off, power, MCS), score them with the
    explicit surrogate, select the top-k and refine them with surrogate-guided coordinate ascent.

    `checkpoint` is a pretrained FM run or an RL run, `surrogate` a surrogate run or an RL run (None: no surrogate).
    """

    def __init__(self, checkpoint='runs/flow_matching', surrogate=None, gen_steps=6, chunk_size=64,
                 score_batch_size=256, refine_batch_size=256, dtype='float32'):
        cfg = load_config(path(checkpoint))
        self.fm = FlowMatching(**{**cfg.model, 'dtype': dtype})
        self.encode_fn, self.decode_fn = load_autoencoder(cfg.autoencoder_path)
        self.params = restore_params(path(checkpoint), 'ema_params')

        # a pretrained FM gets a zero-initialised condition input, i.e., it stays unconditional
        if 'rl' not in cfg:
            edge_dim = 2 * load_config(path(cfg.autoencoder_path)).model.latent_features
            self.params = add_condition_input(self.params, edge_dim)

        self.surrogate, self.s_params = load_surrogate(surrogate, dtype) if surrogate else (None, None)
        self.gen_steps, self.chunk_size = gen_steps, chunk_size
        self.score_fn = jax.jit(partial(surrogate_scores, self.surrogate, batch_size=score_batch_size))
        self.refine = create_refiner(self.surrogate, refine_batch_size)

    def contexts(self, key, scenario, st, n):
        """
        Encoded observations `[n]` of the network with a context axis, each with its own random schedule.
        """

        graphs = []

        for _ in range(n):
            key, tx_key, sim_key = jax.random.split(key, 3)
            graphs.append(observe(tx_key, sim_key, scenario, st))

        contexts = jax.vmap(self.encode_fn)(stack_graphs(graphs))
        return contexts._replace(
            edges=contexts.edges[:, :, None], nodes=contexts.nodes[:, :, None], globals=contexts.globals[:, :, None]
        )

    @partial(jax.jit, static_argnums=0)
    def sample(self, params, key, contexts, cond):
        def single(args):
            context, key, cond = args
            return self.fm.apply(
                params, context, n_steps=self.gen_steps, cond=cond, method='gen', rngs={'sample': key}
            ).edges

        keys = jax.random.split(key, contexts.n_node.shape[0])
        z = jax.lax.map(single, (contexts, keys, cond), batch_size=self.chunk_size)
        return z.astype(contexts.edges.dtype)

    @partial(jax.jit, static_argnums=0)
    def decode(self, contexts, z):
        decode = lambda args: self.decode_fn(latent_graph(*args)).edges
        return jax.lax.map(decode, (contexts, z), batch_size=self.chunk_size)

    @partial(jax.jit, static_argnums=0)
    def decide(self, keys, edges, st, designated):
        return jax.vmap(decide, in_axes=(0, 0, None, 0))(keys, edges, st, designated)

    def generate(self, sample_key, tx_key, params, st, contexts, designated):
        """
        Latents and per-AP decisions `[n, n_aps]` (link or -1, power, MCS) of one sample per context, conditioned on the
        designated (AP, STA) pairs `[n, m, 2]` (-1 = none), which are then also enforced in the decisions.
        """

        cond = jax.vmap(designated_edges)(contexts, designated)
        z = self.sample(params, sample_key, contexts, cond)
        edges = self.decode(contexts, z)
        decisions = self.decide(jax.random.split(tx_key, z.shape[0]), edges, st, designated)
        return z, *(np.asarray(a) for a in decisions)

    def scores(self, s_params, st, choice, power, mcs):
        arrays = choice_to_arrays(st, *map(jnp.asarray, (choice, power, mcs)))
        return np.asarray(self.score_fn(s_params, st, *arrays))
