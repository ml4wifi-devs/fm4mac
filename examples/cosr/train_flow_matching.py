import os
os.environ['JAX_ENABLE_X64'] = 'True'

from functools import partial
from itertools import islice

import hydra
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from examples.cosr.env import observe
from examples.cosr.graph import create_structure, network_graph
from examples.cosr.pipeline import load_autoencoder, path
from fm4mac.data import create_data_loader, load_dataset, save_dataset
from fm4mac.flow import FlowMatching, compute_mmd, context_slots, flow_matching_loss, stack_context
from fm4mac.graphs import make_batch, unbatch
from fm4mac.training import create_optimizer, fit


def encode_dataset(paths, encode_fn, key):
    """
    Training pairs of FM: the latents of a random observation of the network (context) and of a configuration of the
    dataset (target), stacked on axis 1 and stored in float32 to halve the memory.
    """

    items = []

    for p in paths:
        for item in tqdm(load_dataset(path(p)), desc=p):
            scenario = item['scenario']
            st = create_structure(scenario)

            for configuration in item['configurations']:
                key, tx_key, sim_key = jax.random.split(key, 3)
                batch = make_batch([observe(tx_key, sim_key, scenario, st), network_graph(st, *configuration)])
                context, target = unbatch(jax.tree.map(np.array, encode_fn(batch)))
                items.append(cast(stack_context(context, target), np.float32))

    return items


def load_encoded(encoded_path, paths, encode_fn, key):
    if not os.path.exists(path(encoded_path)):
        os.makedirs(os.path.dirname(path(encoded_path)), exist_ok=True)
        save_dataset(encode_dataset(paths, encode_fn, key), path(encoded_path))

    return load_dataset(path(encoded_path))


def cast(graph, dtype):
    return graph._replace(**{name: np.asarray(getattr(graph, name), dtype) for name in ('edges', 'nodes', 'globals')})


@hydra.main(config_path='configs', config_name='flow_matching', version_base=None)
def main(cfg):
    model = FlowMatching(**cfg.model)
    encode_fn, _ = load_autoencoder(cfg.autoencoder_path)
    data_key, val_data_key, init_key, key = jax.random.split(jax.random.PRNGKey(cfg.train.seed), 4)
    items = load_encoded(cfg.encoded_path, cfg.train_paths, encode_fn, data_key)
    val_items = load_encoded(cfg.encoded_val_path, cfg.val_paths, encode_fn, val_data_key)
    to_float64 = partial(cast, dtype=np.float64)
    loader = create_data_loader(items, cfg.train.batch_size, cfg.train.seed, to_float64)
    val_loader = create_data_loader(val_items, cfg.train.batch_size, cfg.train.seed, to_float64)
    val_batches = list(islice(val_loader, cfg.train.val_steps))

    loss_fn = lambda params, key, batch: (flow_matching_loss(model, params, key, batch), {})
    eval_fn = jax.jit(lambda params, key, batch: flow_matching_loss(model, params, key, batch, training=False))

    @jax.jit
    def mmd_fn(params, key, batch):
        mask = jraph.get_edge_padding_mask(batch)
        generate = lambda k: model.apply(params, context_slots(batch), method='gen', rngs={'sample': k}).edges
        mmd = lambda k: compute_mmd(batch.edges[:, -1], generate(k), mask, mask)
        return jax.vmap(mmd)(jax.random.split(key, cfg.train.mmd_samples)).mean()

    def val_fn(params, key):
        keys = jax.random.split(key, len(val_batches))
        loss = np.mean([eval_fn(params, k, b) for k, b in zip(keys, val_batches)])
        return {'loss': loss, 'mmd': mmd_fn(params, key, val_batches[0])}

    t = jnp.zeros((val_batches[0].edges.shape[0], 1))
    params = model.init({'params': init_key, 'dropout': init_key}, val_batches[0], t)
    optimizer = create_optimizer(
        cfg.optimizer.learning_rate, cfg.optimizer.weight_decay, cfg.optimizer.b1, cfg.optimizer.b2,
        cfg.optimizer.grad_clip
    )
    run_dir = path(os.path.join('runs', cfg.name))
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    fit(
        loss_fn, params, optimizer, loader, cfg.train.n_steps, key, run_dir, val_fn, cfg.train.val_freq,
        cfg.train.log_freq, cfg.train.ckpt_freq, cfg.optimizer.ema_decay
    )


if __name__ == '__main__':
    main()
