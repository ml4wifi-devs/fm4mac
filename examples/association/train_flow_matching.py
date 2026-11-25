import os
os.environ['JAX_ENABLE_X64'] = 'True'

from glob import glob

import hydra
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from examples.association.dataset import LOCAL_SEARCH, RANDOM
from examples.association.graph import build_topology, to_graph
from examples.association.pipeline import load_autoencoder, resolve
from fm4mac.data import load_dataset, prebatch
from fm4mac.flow import FlowMatching, compute_mmd, context_slots, flow_matching_loss
from fm4mac.graphs import make_batch, unbatch
from fm4mac.training import create_optimizer, fit


def encode_item(item, rng, encode_fn):
    """
    Encoded (context, target) pairs of one deployment: each local-search association is a target with one random
    association of the same deployment, observed with its throughput, as the context. Latents are stacked as
    [n, context + target, d].
    """

    targets = np.flatnonzero(item.source == LOCAL_SEARCH)

    if len(targets) == 0:
        return []

    contexts = rng.choice(np.flatnonzero(item.source == RANDOM), len(targets))
    topology, primary = build_topology(item.deployment), np.asarray(item.deployment.primary)
    idxs = np.concatenate([contexts, targets])
    graphs = [to_graph(topology, primary, item.plans[j], item.throughput[j], xp=np) for j in idxs]
    encoded = unbatch(encode_fn(make_batch(graphs)))
    stack = lambda c, t: np.stack([c, t], axis=1)
    return [
        t._replace(edges=stack(c.edges, t.edges), nodes=stack(c.nodes, t.nodes), globals=stack(c.globals, t.globals))
        for c, t in zip(encoded[:len(targets)], encoded[len(targets):])
    ]


def encoded_batches(pattern, encode_fn, max_edges, n_batches, rng):
    items = sum((load_dataset(p) for p in sorted(glob(pattern))), start=[])
    samples = [s for item in tqdm(items, desc='Encoding') for s in encode_item(item, rng, encode_fn)]
    return prebatch(samples, max_edges, n_batches, rng)


@hydra.main(config_path='configs', config_name='flow_matching', version_base=None)
def main(cfg):
    run_dir = resolve(os.path.join('runs', cfg.name))
    train = cfg.train
    encode_fn, _ = load_autoencoder(cfg.autoencoder_path)
    rng = np.random.default_rng(train.seed)
    pattern = lambda split: resolve(os.path.join(cfg.dataset_path, f'{split}_*.pkl.lz4'))
    data = encoded_batches(pattern('train'), encode_fn, train.max_edges, train.n_steps, rng)
    val_data = encoded_batches(pattern('val'), encode_fn, train.max_edges, train.n_val_batches, rng)
    model = FlowMatching(**cfg.model)

    def loss_fn(params, key, batch):
        return flow_matching_loss(model, params, key, batch), {}

    @jax.jit
    def val_step(params, key, batch):
        loss_key, gen_key = jax.random.split(key)
        generated = model.apply(params, context_slots(batch), method='gen', rngs={'sample': gen_key})
        mask = jraph.get_edge_padding_mask(batch)
        mmd = compute_mmd(batch.edges[:, -1], generated.edges, mask, mask)
        return {'loss': flow_matching_loss(model, params, loss_key, batch, training=False), 'mmd': mmd}

    def val_fn(params, key):
        idxs = rng.choice(len(val_data), train.val_batches)
        metrics = [val_step(params, k, val_data[i]) for k, i in zip(jax.random.split(key, len(idxs)), idxs)]
        return jax.tree.map(lambda *x: jnp.mean(jnp.asarray(x)), *metrics)

    init_key, key = jax.random.split(jax.random.PRNGKey(train.seed))
    params = model.init({'params': init_key, 'dropout': init_key}, data[0], jnp.zeros((data[0].edges.shape[0], 1)))
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    optimizer = create_optimizer(**cfg.optimizer)
    fit(
        loss_fn, params, optimizer, data, train.n_steps, key, run_dir, val_fn, train.val_freq, train.log_freq,
        train.ckpt_freq, train.ema_decay
    )


if __name__ == '__main__':
    main()
