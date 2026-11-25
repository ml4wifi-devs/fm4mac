import os
os.environ['JAX_ENABLE_X64'] = 'True'

from glob import glob

import hydra
import jax
import numpy as np
from omegaconf import OmegaConf

from examples.association.graph import build_topology, plan_edges, plan_graph, sta_targets, target_mask
from examples.association.pipeline import resolve
from fm4mac.data import load_dataset
from fm4mac.graphs import nearest_power_of_two, pad_graph
from fm4mac.search import rank_correlation
from fm4mac.surrogate import Surrogate, surrogate_loss
from fm4mac.training import create_optimizer, fit, one_cycle


def load_items(pattern):
    items = sum((load_dataset(p) for p in sorted(glob(pattern))), start=[])
    return items, [build_topology(item.deployment) for item in items]


def group_batch(item, topology, idxs):
    """
    Padded explicit graphs of a group of associations of one deployment (stacked on the first axis), their per-edge
    targets and target masks, and group ids.
    """

    primary = np.asarray(item.deployment.primary)
    graph = plan_graph(topology, primary, item.plans[idxs[0]])
    graph = graph._replace(
        edges=np.stack([plan_edges(topology, primary, item.plans[j]) for j in idxs]),
        nodes=np.zeros((len(idxs),) + graph.nodes.shape, dtype=graph.nodes.dtype)
    )
    n_node_pad, n_edge_pad = nearest_power_of_two(topology.n_nodes + 1), nearest_power_of_two(len(topology.senders))
    graph = pad_graph(graph, n_node_pad, n_edge_pad, np)
    mask = np.stack([target_mask(topology, item.plans[j]) for j in idxs])
    return graph, sta_targets(topology, item.throughput[idxs]).astype(np.float32), mask, np.zeros(len(idxs), dtype=int)


def random_groups(items, topologies, group_size, rng):
    """
    Infinite stream of groups of `group_size` random associations of a random deployment.
    """

    while True:
        i = rng.integers(len(items))
        yield group_batch(items[i], topologies[i], rng.choice(len(items[i].plans), group_size, replace=False))


@hydra.main(config_path='configs', config_name='surrogate', version_base=None)
def main(cfg):
    run_dir = resolve(os.path.join('runs', cfg.name))
    train = cfg.train
    rng = np.random.default_rng(train.seed)
    pattern = lambda split: resolve(os.path.join(cfg.dataset_path, f'{split}_*.pkl.lz4'))
    data = random_groups(*load_items(pattern('train')), train.group_size, rng)
    val_items, val_topologies = load_items(pattern('val'))
    val_data = [
        group_batch(item, topology, idxs) for item, topology in zip(val_items, val_topologies)
        for idxs in np.arange(len(item.plans) // train.group_size * train.group_size).reshape(-1, train.group_size)
    ]
    model = Surrogate(**cfg.model)

    def loss_fn(params, key, batch):
        loss, (nll, rank, _, _) = surrogate_loss(model, params, *batch, train.rank_weight, key)
        return loss, {'nll': nll, 'rank': rank}

    eval_fn = jax.jit(lambda params, batch: surrogate_loss(model, params, *batch, train.rank_weight))

    def val_fn(params, key):
        metrics = [eval_fn(params, batch)[1] for batch in val_data]
        return {
            'nll': np.mean([m[0] for m in metrics]), 'rank': np.mean([m[1] for m in metrics]),
            'spearman': np.nanmean([rank_correlation(np.asarray(m[2])[None], np.asarray(m[3])[None]) for m in metrics])
        }

    init_key, key = jax.random.split(jax.random.PRNGKey(train.seed))
    graph = val_data[0][0]
    params = model.init(init_key, graph._replace(edges=graph.edges[0], nodes=graph.nodes[0]), training=False)
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    schedule = one_cycle(cfg.optimizer.learning_rate, train.n_steps)
    optimizer = create_optimizer(**{**cfg.optimizer, 'learning_rate': schedule})
    fit(
        loss_fn, params, optimizer, data, train.n_steps, key, run_dir, val_fn, train.val_freq, train.log_freq,
        train.ckpt_freq
    )


if __name__ == '__main__':
    main()
