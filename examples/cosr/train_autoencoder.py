import os
os.environ['JAX_ENABLE_X64'] = 'True'

from itertools import islice

import hydra
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from omegaconf import OmegaConf

from examples.cosr.graph import Connection, create_structure, network_graph
from examples.cosr.pipeline import path
from fm4mac.autoencoder import (
    Autoencoder,
    autoencoder_loss,
    binary_cross_entropy,
    mixture_accuracy,
    mixture_categorical_loss,
    squared_error
)
from fm4mac.data import create_data_loader, load_dataset
from fm4mac.graphs import head_sizes, is_categorical
from fm4mac.training import create_optimizer, fit


def load_graphs(paths):
    """
    Network graphs of all configurations of the datasets.
    """

    return [
        network_graph(st, *configuration)
        for p in paths for item in load_dataset(path(p)) for st in [create_structure(item['scenario'])]
        for configuration in item['configurations']
    ]


@hydra.main(config_path='configs', config_name='autoencoder', version_base=None)
def main(cfg):
    names = list(head_sizes(Connection))
    field_losses = {
        name: mixture_categorical_loss if is_categorical(Connection, name)
        else binary_cross_entropy if name == 'success' else squared_error
        for name in names
    }
    weights = np.asarray(cfg.train.loss_weights) / np.sum(cfg.train.loss_weights)
    model = Autoencoder(Connection, **cfg.model)

    def loss_fn(params, key, batch):
        pred, z_mean, z_log_var = model.apply(params, batch, rngs={'sample': key})
        loss, kl, losses = autoencoder_loss(pred, batch, z_mean, z_log_var, field_losses, weights, cfg.train.kl_weight)
        mask = jraph.get_edge_padding_mask(batch)
        metrics = {'kl': kl} | {f'loss/{name}': loss for name, loss in zip(names, losses)}

        for name in names:
            if is_categorical(Connection, name):
                _, acc, applicable = mixture_accuracy(getattr(pred.edges, name), getattr(batch.edges, name))
                metrics[f'acc/{name}'] = (mask * acc).sum() / jnp.maximum((mask * applicable).sum(), 1)

        return loss, metrics

    loader = create_data_loader(load_graphs(cfg.train_paths), cfg.train.batch_size, cfg.train.seed)
    val_loader = create_data_loader(load_graphs(cfg.val_paths), cfg.train.batch_size, cfg.train.seed)
    val_batches = list(islice(val_loader, cfg.train.val_steps))
    eval_fn = jax.jit(loss_fn)

    def val_fn(params, key):
        out = [eval_fn(params, k, b) for k, b in zip(jax.random.split(key, len(val_batches)), val_batches)]
        return jax.tree.map(lambda *x: np.mean(x), *[{'loss': loss} | metrics for loss, metrics in out])

    init_key, key = jax.random.split(jax.random.PRNGKey(cfg.train.seed))
    params = model.init({'params': init_key, 'sample': init_key}, val_batches[0])
    optimizer = create_optimizer(
        cfg.optimizer.learning_rate, cfg.optimizer.weight_decay, cfg.optimizer.b1, cfg.optimizer.b2,
        cfg.optimizer.grad_clip
    )
    run_dir = path(os.path.join('runs', cfg.name))
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    fit(
        loss_fn, params, optimizer, loader, cfg.train.n_steps, key, run_dir, val_fn, cfg.train.val_freq,
        cfg.train.log_freq, cfg.train.ckpt_freq
    )


if __name__ == '__main__':
    main()
