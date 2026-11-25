import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import itertools

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from examples.channel.dataset import LOCAL_SEARCH, RANDOM
from examples.channel.env import decode_action
from examples.channel.graph import build_topology, to_graph
from examples.channel.pipeline import load_autoencoder, resolve
from examples.channel.train_autoencoder import load_items
from fm4mac.data import prebatch
from fm4mac.flow import FlowMatching, flow_matching_loss, stack_context
from fm4mac.graphs import make_batch, unbatch
from fm4mac.training import create_optimizer, fit


def encode_item(item, rng, encode_fn):
    """
    Encoded (context, target) pairs: each local-search plan is a target and gets one random plan of the same
    deployment, observed with its throughput, as the context.
    """

    targets = np.flatnonzero(item['source'] == LOCAL_SEARCH)
    contexts = rng.choice(np.flatnonzero(item['source'] == RANDOM), len(targets))
    topology = build_topology(item['ap_pos'], item['sta_pos'], item['sta_mask'])
    graphs = [
        to_graph(topology, *decode_action(item['plans'][j]), item['throughput'][j], xp=np)
        for j in np.concatenate([contexts, targets])
    ]
    encoded = unbatch(jax.tree.map(np.array, encode_fn(make_batch(graphs))))
    n = len(targets)
    return [jax.tree.map(np.asarray, stack_context(c, t)) for c, t in zip(encoded[:n], encoded[n:])]


def encode_dataset(pattern, rng, encode_fn):
    return [s for item in tqdm(load_items(pattern), desc=pattern) for s in encode_item(item, rng, encode_fn)]


@hydra.main(config_path='configs', config_name='flow_matching', version_base=None)
def main(cfg):
    encode_fn, _ = load_autoencoder(cfg.autoencoder_path)
    rng = np.random.default_rng(cfg.train.seed)
    samples = encode_dataset(cfg.dataset, rng, encode_fn)
    val_samples = encode_dataset(cfg.val_dataset, rng, encode_fn)
    val_batches = prebatch(val_samples, cfg.train.max_edges, cfg.train.n_val_batches, rng)
    batches = itertools.chain.from_iterable(prebatch(samples, cfg.train.max_edges, 100, rng) for _ in itertools.count())

    model = FlowMatching(**cfg.model)
    loss_fn = lambda params, key, batch: (flow_matching_loss(model, params, key, batch), {})
    val_loss = jax.jit(lambda params, key, batch: flow_matching_loss(model, params, key, batch, training=False))
    val_fn = lambda params, key: dict(loss=jnp.mean(jnp.asarray([
        val_loss(params, k, b) for k, b in zip(jax.random.split(key, len(val_batches)), val_batches)
    ])))

    init_key, key = jax.random.split(jax.random.PRNGKey(cfg.train.seed))
    sample = val_batches[0]
    params = model.init({'params': init_key, 'dropout': init_key}, sample, jnp.zeros((sample.edges.shape[0], 1)))

    run_dir = resolve(os.path.join('runs', cfg.name))
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    fit(
        loss_fn, params, create_optimizer(**cfg.optimizer), batches, cfg.train.n_steps, key, run_dir,
        val_fn=val_fn, val_freq=cfg.train.val_freq, log_freq=cfg.train.log_freq, ckpt_freq=cfg.train.ckpt_freq,
        ema_decay=cfg.train.ema_decay
    )


if __name__ == '__main__':
    main()
