import os
os.environ['JAX_ENABLE_X64'] = 'True'

from dataclasses import fields as dataclass_fields
from glob import glob

import hydra
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from omegaconf import OmegaConf

from examples.association.graph import AssocEdge, build_topology, to_graph
from examples.association.pipeline import resolve
from fm4mac.autoencoder import (
    Autoencoder,
    autoencoder_loss,
    mixture_accuracy,
    mixture_categorical_loss,
    squared_error
)
from fm4mac.data import create_data_loader, load_dataset
from fm4mac.graphs import is_categorical
from fm4mac.training import create_optimizer, fit


def association_graphs(pattern):
    """
    Lazy autoencoder graphs of all (deployment, association) pairs of the datasets matching `pattern`.
    """

    items = sum((load_dataset(p) for p in sorted(glob(pattern))), start=[])
    topologies = [build_topology(item.deployment) for item in items]
    pairs = [(i, j) for i, item in enumerate(items) for j in range(len(item.plans))]

    def graph(pair):
        i, j = pair
        primary = np.asarray(items[i].deployment.primary)
        return to_graph(topologies[i], primary, items[i].plans[j], items[i].throughput[j], xp=np)

    return pairs, graph


@hydra.main(config_path='configs', config_name='autoencoder', version_base=None)
def main(cfg):
    run_dir = resolve(os.path.join('runs', cfg.name))
    train = cfg.train
    fields = [f.name for f in dataclass_fields(AssocEdge)]
    losses = {f: mixture_categorical_loss if is_categorical(AssocEdge, f) else squared_error for f in fields}
    weights = np.asarray(train.loss_weights) / np.sum(train.loss_weights)
    model = Autoencoder(AssocEdge, **cfg.model)

    def loss_fn(params, key, batch):
        pred, z_mean, z_log_var = model.apply(params, batch, rngs={'sample': key})
        loss, kl, field_losses = autoencoder_loss(pred, batch, z_mean, z_log_var, losses, weights, train.kl_weight)
        mask = jraph.get_edge_padding_mask(pred)
        na_acc, acc, applicable = mixture_accuracy(pred.edges.assoc, batch.edges.assoc)
        metrics = {'kl': kl, 'assoc_acc': (mask * acc).sum() / (mask * applicable).sum()}
        return loss, metrics | {f'loss/{f}': v for f, v in zip(fields, field_losses)}

    pairs, graph = association_graphs(resolve(os.path.join(cfg.dataset_path, 'train_*.pkl.lz4')))
    val_pairs, val_graph = association_graphs(resolve(os.path.join(cfg.dataset_path, 'val_*.pkl.lz4')))
    data = create_data_loader(pairs, train.batch_size, train.seed, graph)
    val_data = iter(create_data_loader(val_pairs, train.batch_size, train.seed, val_graph))
    eval_fn = jax.jit(loss_fn)

    def val_fn(params, key):
        metrics = [eval_fn(params, k, next(val_data)) for k in jax.random.split(key, train.val_batches)]
        return jax.tree.map(lambda *x: jnp.mean(jnp.asarray(x)), *[{'loss': loss} | m for loss, m in metrics])

    init_key, key = jax.random.split(jax.random.PRNGKey(train.seed))
    params = model.init({'params': init_key, 'sample': init_key}, next(iter(data)))
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    optimizer = create_optimizer(**cfg.optimizer)
    fit(
        loss_fn, params, optimizer, data, train.n_steps, key, run_dir, val_fn, train.val_freq, train.log_freq,
        train.ckpt_freq
    )


if __name__ == '__main__':
    main()
