import os
os.environ['JAX_ENABLE_X64'] = 'True'

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from examples.channel.env import N_ACTIONS, decode_action, encode_action, evaluate, random_plan
from examples.channel.graph import ChannelEdge, plan_graph, sample_plan, to_graph
from fm4mac.autoencoder import Autoencoder
from fm4mac.checkpoints import load_config, restore_params
from fm4mac.flow import FlowMatching, latent_graph
from fm4mac.graphs import stack_graphs
from fm4mac.search import create_coordinate_ascent
from fm4mac.surrogate import Surrogate, utility


ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    """
    Paths are relative to the example directory (absolute paths are kept).
    """

    return os.path.join(ROOT, path)


def load_autoencoder(path):
    cfg = load_config(resolve(path))
    model = Autoencoder(ChannelEdge, **cfg.model)
    params = restore_params(resolve(path))
    return tuple(jax.jit(partial(model.apply, params, method=method)) for method in ('encode', 'decode'))


def load_flow_matching(path):
    """
    Flow matching model, its parameters (EMA of a flow matching run or the fine-tuned ones of an RL run) and the
    autoencoder path.
    """

    cfg = load_config(resolve(path))

    if 'flow_matching_path' in cfg:
        model, _, autoencoder_path = load_flow_matching(cfg.flow_matching_path)
        return model, restore_params(resolve(path)), autoencoder_path

    return FlowMatching(**cfg.model), restore_params(resolve(path), 'ema_params'), cfg.autoencoder_path


def load_surrogate(path):
    """
    Explicit plan surrogate and its parameters (of a surrogate run or the fine-tuned ones of an RL run).
    """

    cfg = load_config(resolve(path))

    if 'surrogate_path' in cfg:
        return load_surrogate(cfg.surrogate_path)[0], restore_params(resolve(path), 'surrogate_params')

    return Surrogate(**cfg.model, embed_nodes=False), restore_params(resolve(path))


def observe(key, deployment, topology, encode_fn, n_contexts):
    """
    Encoded observations of random plans with their simulated throughput, as flow matching contexts stacked on
    axis 0 (one context slot).
    """

    plan_key, sim_key = jax.random.split(key)
    plans = jnp.stack([random_plan(k, topology.n_aps) for k in jax.random.split(plan_key, n_contexts)])
    thr = evaluate(jax.random.split(sim_key, n_contexts), *deployment, plans)
    contexts = jax.vmap(encode_fn)(stack_graphs([to_graph(topology, *decode_action(p), t) for p, t in zip(plans, thr)]))
    return contexts._replace(
        edges=contexts.edges[:, :, None], nodes=contexts.nodes[:, :, None], globals=contexts.globals[:, :, None]
    )


def create_sampler(model, decode_fn, n_aps, chunk_size, n_steps=6):
    """
    Jitted sampling of one latent configuration and its decoded channel plan per context (stacked on axis 0), in
    chunks of `chunk_size` contexts.
    """

    @jax.jit
    def sample(params, gen_key, plan_key, contexts):
        def single(args):
            context, gen_key, plan_key = args
            z = model.apply(params, context, n_steps=n_steps, method='gen', rngs={'sample': gen_key}).edges
            z = z.astype(context.edges.dtype)
            return z, encode_action(*sample_plan(plan_key, decode_fn(latent_graph(context, z)), n_aps))

        n = contexts.n_node.shape[0]
        keys = (jax.random.split(gen_key, n), jax.random.split(plan_key, n))
        return jax.lax.map(single, (contexts, *keys), batch_size=chunk_size)

    return sample


def create_scorer(model, topology, chunk_size):
    """
    Jitted surrogate scores (predicted PF utility) of plans `[n, n_aps]`, in chunks of `chunk_size` plans.
    """

    mask = jnp.ones(topology.n_aps)

    @jax.jit
    def score(params, plans):
        def single(plan):
            mean, _ = model.apply(params, plan_graph(topology, *decode_action(plan), xp=jnp), training=False)
            return utility(mean, mask)

        return jax.lax.map(single, plans, batch_size=chunk_size)

    return score


def create_refiner(score_fn, n_aps):
    """
    Surrogate-guided coordinate ascent over the 32 actions of each AP (no simulator calls).
    """

    search = create_coordinate_ascent(lambda params, c: score_fn(params, c.reshape(-1, n_aps)).reshape(c.shape[:2]))
    options = np.tile(np.arange(N_ACTIONS), (n_aps, 1))
    return lambda params, plans, rng, n_rounds=2: search(params, plans, options, True, rng, n_rounds)
