import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from examples.channel.env import decode_action
from examples.channel.graph import build_topology, plan_graph
from examples.channel.pipeline import resolve
from examples.channel.train_autoencoder import load_items
from fm4mac.graphs import nearest_power_of_two, pad_graph
from fm4mac.search import rank_correlation
from fm4mac.surrogate import Surrogate, surrogate_loss
from fm4mac.training import create_optimizer, fit, one_cycle


def plan_group(item, idxs):
    """
    Padded explicit graphs of plans `idxs` of one deployment with the per-AP log-throughput targets on the AP
    self-loops (first edges) and their mask; the plans form one ranking group.
    """

    topology = build_topology(item['ap_pos'], item['sta_pos'], item['sta_mask'])
    n_edge_pad = nearest_power_of_two(len(topology.senders))
    graphs = pad_graph(
        plan_graph(topology, *decode_action(item['plans'][idxs])), nearest_power_of_two(topology.n_nodes + 1),
        n_edge_pad, xp=np
    )
    targets = np.zeros((len(idxs), n_edge_pad), dtype=np.float32)
    targets[:, :topology.n_aps] = np.log(np.maximum(item['throughput'][idxs], 1.))
    mask = np.zeros_like(targets)
    mask[:, :topology.n_aps] = 1.
    return graphs, targets, mask, np.zeros(len(idxs), dtype=np.int32)


def random_groups(items, group_size, rng):
    while True:
        item = items[rng.integers(len(items))]
        yield plan_group(item, rng.choice(len(item['plans']), group_size, replace=False))


@hydra.main(config_path='configs', config_name='surrogate', version_base=None)
def main(cfg):
    model = Surrogate(**cfg.model, embed_nodes=False)
    rank_weight = cfg.train.rank_weight
    rng = np.random.default_rng(cfg.train.seed)
    data = random_groups(load_items(cfg.dataset), cfg.train.group_size, rng)
    val_items = load_items(cfg.val_dataset)
    val_data = [plan_group(it, rng.choice(len(it['plans']), cfg.train.group_size, replace=False)) for it in val_items]

    def loss_fn(params, key, batch):
        loss, (nll, rank, _, _) = surrogate_loss(model, params, *batch, rank_weight, key)
        return loss, dict(nll=nll, rank=rank)

    eval_fn = jax.jit(lambda params, batch: surrogate_loss(model, params, *batch, rank_weight)[1])

    def val_fn(params, _):
        nll, rank, pred, true = zip(*[eval_fn(params, batch) for batch in val_data])
        corr = np.nanmean([rank_correlation(np.asarray(p)[None], np.asarray(t)[None]) for p, t in zip(pred, true)])
        return dict(nll=jnp.mean(jnp.asarray(nll)), rank=jnp.mean(jnp.asarray(rank)), spearman=corr)

    init_key, key = jax.random.split(jax.random.PRNGKey(cfg.train.seed))
    item = val_items[0]
    topology = build_topology(item['ap_pos'], item['sta_pos'], item['sta_mask'])
    params = model.init(init_key, plan_graph(topology, *decode_action(item['plans'][0])), training=False)

    run_dir = resolve(os.path.join('runs', cfg.name))
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    opt = cfg.optimizer
    lr = one_cycle(opt.learning_rate, cfg.train.n_steps)
    optimizer = create_optimizer(lr, opt.weight_decay, opt.b1, opt.b2, opt.grad_clip)
    fit(
        loss_fn, params, optimizer, data, cfg.train.n_steps, key, run_dir,
        val_fn=val_fn, val_freq=cfg.train.val_freq, log_freq=cfg.train.log_freq, ckpt_freq=cfg.train.n_steps
    )


if __name__ == '__main__':
    main()
