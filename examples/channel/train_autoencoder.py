import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

from glob import glob

import hydra
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from omegaconf import OmegaConf

from examples.channel.env import decode_action
from examples.channel.graph import ChannelEdge, build_topology, to_graph
from examples.channel.pipeline import resolve
from fm4mac.autoencoder import Autoencoder, autoencoder_loss, mixture_categorical_loss, squared_error
from fm4mac.data import create_data_loader, load_dataset
from fm4mac.training import create_optimizer, fit


FIELD_LOSSES = dict(
    primary=mixture_categorical_loss, rssi=squared_error, throughput=squared_error,
    type=mixture_categorical_loss, width=mixture_categorical_loss
)


def load_items(pattern):
    return sum((load_dataset(p) for p in sorted(glob(resolve(pattern)))), start=[])


def plan_loader(items, batch_size, seed):
    """
    Infinite loader of batches of (deployment, plan) graphs with the simulated AP throughput.
    """

    topologies = [build_topology(it['ap_pos'], it['sta_pos'], it['sta_mask']) for it in items]
    index = [(i, j) for i, it in enumerate(items) for j in range(len(it['plans']))]
    graph = lambda ij: to_graph(
        topologies[ij[0]], *decode_action(items[ij[0]]['plans'][ij[1]]), items[ij[0]]['throughput'][ij[1]], xp=np
    )
    return create_data_loader(index, batch_size, seed, map_fn=graph)


def plan_accuracy(pred, batch):
    """
    Fraction of AP self-loops with both the primary channel and the width reconstructed.
    """

    is_plan = jraph.get_edge_padding_mask(batch) & (batch.edges.width[:, -1] == 0)
    same = lambda name: getattr(pred.edges, name)[:, :-1].argmax(-1) == getattr(batch.edges, name)[:, :-1].argmax(-1)
    return (is_plan & same('primary') & same('width')).sum() / is_plan.sum()


@hydra.main(config_path='configs', config_name='autoencoder', version_base=None)
def main(cfg):
    model = Autoencoder(ChannelEdge, **cfg.model)
    weights = np.asarray(cfg.train.loss_weights) / np.sum(cfg.train.loss_weights)

    def loss_fn(params, key, batch):
        pred, z_mean, z_log_var = model.apply(params, batch, rngs={'sample': key})
        loss, kl, _ = autoencoder_loss(pred, batch, z_mean, z_log_var, FIELD_LOSSES, weights, cfg.train.kl_weight)
        return loss, dict(kl=kl, plan_acc=plan_accuracy(pred, batch))

    @jax.jit
    def eval_fn(params, key, batch):
        loss, metrics = loss_fn(params, key, batch)
        return metrics | dict(loss=loss)

    def val_fn(params, key):
        metrics = [eval_fn(params, k, next(val_data)) for k in jax.random.split(key, cfg.train.n_val_batches)]
        return jax.tree.map(lambda *x: jnp.mean(jnp.asarray(x)), *metrics)

    data = iter(plan_loader(load_items(cfg.dataset), cfg.train.batch_size, cfg.train.seed))
    val_data = iter(plan_loader(load_items(cfg.val_dataset), cfg.train.batch_size, cfg.train.seed))
    init_key, key = jax.random.split(jax.random.PRNGKey(cfg.train.seed))
    params = model.init({'params': init_key, 'sample': init_key}, next(val_data))

    run_dir = resolve(os.path.join('runs', cfg.name))
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    optimizer = create_optimizer(**cfg.optimizer)
    fit(
        loss_fn, params, optimizer, data, cfg.train.n_steps, key, run_dir,
        val_fn=val_fn, val_freq=cfg.train.val_freq, log_freq=cfg.train.log_freq, ckpt_freq=cfg.train.ckpt_freq
    )


if __name__ == '__main__':
    main()
