import os
os.environ['JAX_ENABLE_X64'] = 'True'

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from examples.association.env import evaluate, random_assoc
from examples.association.graph import (
    AssocEdge,
    assoc_scores,
    plan_edges,
    plan_graph,
    sample_assoc,
    target_mask,
    to_graph
)
from fm4mac.autoencoder import Autoencoder
from fm4mac.checkpoints import load_config, restore_params
from fm4mac.flow import FlowMatching, latent_graph
from fm4mac.graphs import stack_graphs
from fm4mac.search import create_coordinate_ascent
from fm4mac.surrogate import Surrogate, utility


ROOT = os.path.dirname(os.path.abspath(__file__))
SURROGATE_PATH = 'runs/surrogate'


def resolve(path):
    """
    Paths relative to the example directory (absolute paths are kept).
    """

    return os.path.join(ROOT, path)


def load_autoencoder(path):
    """
    Jitted encode and decode functions of an autoencoder run.
    """

    model = Autoencoder(AssocEdge, **load_config(resolve(path)).model)
    params = restore_params(resolve(path))
    encode_fn, decode_fn = (jax.jit(partial(model.apply, params, method=method)) for method in ('encode', 'decode'))
    return encode_fn, decode_fn


def load_flow_matching(path):
    """
    FM config and parameters of a pretrained FM run (EMA parameters) or of an RL run (fine-tuned parameters).
    """

    cfg = load_config(resolve(path))

    if 'flow_matching_path' in cfg:
        return load_config(resolve(cfg.flow_matching_path)), restore_params(resolve(path), 'params')

    return cfg, restore_params(resolve(path), 'ema_params')


def load_surrogate(path):
    """
    Surrogate model and parameters of a surrogate run or the fine-tuned surrogate of an RL run.
    """

    cfg = load_config(resolve(path))

    if 'surrogate_path' in cfg:
        model = Surrogate(**load_config(resolve(cfg.surrogate_path)).model)
        return model, restore_params(resolve(path), 'surrogate_params')

    return Surrogate(**cfg.model), restore_params(resolve(path))


def load_pipeline(run, surrogate_path=None):
    """
    FM model and parameters, autoencoder functions, and surrogate model and parameters of a pretrained FM run or an
    RL run. The surrogate defaults to the fine-tuned one of an RL run or `SURROGATE_PATH`.
    """

    surrogate_path = surrogate_path or (run if 'flow_matching_path' in load_config(resolve(run)) else SURROGATE_PATH)
    cfg, params = load_flow_matching(run)
    return FlowMatching(**cfg.model), params, *load_autoencoder(cfg.autoencoder_path), *load_surrogate(surrogate_path)


def create_contexts(key, net, topology, primary, encode_fn, n_contexts):
    """
    Encoded observations of `n_contexts` random associations with their simulated throughput (the FM condition),
    stacked on the first axis with a context slot on axis 2.
    """

    plan_key, sim_key = jax.random.split(key)
    rng = np.random.default_rng(np.asarray(plan_key))
    plans = np.stack([random_assoc(rng, net) for _ in range(n_contexts)])
    thr = evaluate(jax.random.split(sim_key, n_contexts), net, jnp.asarray(plans))
    graphs = [to_graph(topology, primary, jnp.asarray(p), t) for p, t in zip(plans, thr)]
    contexts = jax.vmap(encode_fn)(stack_graphs(graphs))
    return contexts._replace(
        edges=contexts.edges[:, :, None], nodes=contexts.nodes[:, :, None], globals=contexts.globals[:, :, None]
    )


def create_sampler(fm, n_steps=6, chunk_size=128):
    """
    FM latents of stacked contexts, `chunk_size` contexts at a time.
    """

    @jax.jit
    def sample(params, key, contexts):
        def single(args):
            context, key = args
            return fm.apply(params, context, n_steps=n_steps, method='gen', rngs={'sample': key}).edges

        keys = jax.random.split(key, contexts.n_node.shape[0])
        return jax.lax.map(single, (contexts, keys), batch_size=chunk_size).astype(contexts.edges.dtype)

    return sample


def create_decoder(decode_fn, topology, chunk_size=128):
    """
    Associations sampled from the decoded association logits of FM latents.
    """

    @jax.jit
    def decode(keys, contexts, z):
        def single(args):
            key, context, z = args
            scores = assoc_scores(decode_fn(latent_graph(context, z)).edges.assoc, topology)
            return sample_assoc(key, scores, topology)

        return jax.lax.map(single, (keys, contexts, z), batch_size=chunk_size)

    return decode


def plan_graphs(topology, primary, plans):
    """
    Explicit association graphs of a batch of associations, stacked for `fm4mac.surrogate.apply_batch`, and their
    target masks.
    """

    graph = plan_graph(topology, np.asarray(primary), np.zeros(topology.n_stas, dtype=int))
    graphs = graph._replace(
        nodes=jnp.zeros((len(plans),) + graph.nodes.shape, dtype=graph.nodes.dtype),
        edges=jax.vmap(lambda p: plan_edges(topology, primary, p, jnp))(plans)
    )
    return graphs, jax.vmap(lambda p: target_mask(topology, p, jnp))(plans)


def create_scorer(surrogate, topology, primary, chunk_size=128):
    """
    Predicted PF utility (sum of the per-STA log-throughput predictions) of a batch of associations.
    """

    @jax.jit
    def score(s_params, plans):
        def single(plan):
            mean, _ = surrogate.apply(s_params, plan_graph(topology, primary, plan, xp=jnp), training=False)
            return utility(mean, target_mask(topology, plan, xp=jnp))

        return jax.lax.map(single, plans, batch_size=chunk_size)

    return score


def create_refiner(score):
    """
    Surrogate-guided refinement: STA-wise coordinate ascent over the candidate APs (`env.option_table`) with the
    surrogate as the scorer. Call with `args=s_params`.
    """

    def score_options(s_params, candidates):
        return score(s_params, candidates.reshape(-1, candidates.shape[-1])).reshape(candidates.shape[:2])

    return create_coordinate_ascent(score_options)
