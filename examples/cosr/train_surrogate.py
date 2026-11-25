import os
os.environ['JAX_ENABLE_X64'] = 'True'

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from examples.cosr.env import create_simulator, observe, random_choice
from examples.cosr.graph import LINK_SCALE, choice_to_arrays, create_structure, explicit_graph
from examples.cosr.pipeline import Pipeline, path
from fm4mac.data import load_dataset, save_dataset
from fm4mac.graphs import make_batch, nearest_power_of_two, pad_graph, stack_graphs
from fm4mac.search import rank_correlation
from fm4mac.surrogate import Surrogate, surrogate_loss
from fm4mac.training import create_optimizer, fit, one_cycle


SOURCES = {'fm': 0, 'fm_designated': 1, 'random': 2, 'perturbed': 3}


def perturb(rng, st, choice, power, mcs, n):
    """
    Single-AP moves of random base decisions: switch off, change the station, the power, the MCS, or all of them.
    """

    base = rng.integers(0, len(choice), n)
    choice, power, mcs = choice[base].copy(), power[base].copy(), mcs[base].copy()
    _, new_power, new_mcs = random_choice(rng, st, n)
    n_options = (st.ap_links >= 0).sum(axis=-1)

    for i in range(n):
        ap = rng.integers(len(st.ap_ids))
        move = rng.integers(5) if choice[i, ap] >= 0 else 4

        if move in (1, 4):
            choice[i, ap] = st.ap_links[ap, rng.integers(n_options[ap])]
        if move == 0:
            choice[i, ap] = -1
        if move in (2, 4):
            power[i, ap] = new_power[i, ap]
        if move in (3, 4):
            mcs[i, ap] = new_mcs[i, ap]

    return choice, power, mcs


def random_designated(rng, st, n, max_designated=3):
    """
    Up to `max_designated` (AP, STA) pairs of random APs for the second half of `n` samples (-1 = none).
    """

    n_aps = len(st.ap_ids)
    designated = np.full((n, max_designated, 2), -1)

    for i in range(n // 2, n):
        aps = rng.permutation(n_aps)[:rng.integers(1, min(max_designated, n_aps) + 1)]
        links = [st.ap_links[a, rng.integers((st.ap_links[a] >= 0).sum())] for a in aps]
        designated[i, :len(links)] = np.stack([st.link_ap[links], st.link_sta[links]], axis=-1)

    return designated


def generate(cfg, paths, n_networks, seed):
    """
    Groups of schedules of random training networks: FM samples (half of them with designated stations) given one
    observation, random decisions and single-AP perturbations of the FM samples, with simulated link rates.
    """

    pipe = Pipeline(cfg.flow_matching_path, gen_steps=cfg.gen_steps, chunk_size=cfg.n_fm)
    scenarios = [item['scenario'] for p in paths for item in load_dataset(path(p))]
    idxs = np.sort(np.random.default_rng(seed).permutation(len(scenarios))[:n_networks])
    key, rng = jax.random.PRNGKey(seed), np.random.default_rng(seed)
    groups = []

    for scenario in tqdm([scenarios[i] for i in idxs], desc='Networks'):
        st = create_structure(scenario)
        key, tx_key, sim_key, sample_key, decode_key, eval_key = jax.random.split(key, 6)
        context = pipe.encode_fn(make_batch([observe(tx_key, sim_key, scenario, st)]))
        context = context._replace(edges=context.edges[:, None], nodes=context.nodes[:, None],
                                   globals=context.globals[:, None])
        designated = random_designated(rng, st, cfg.n_fm)
        contexts = stack_graphs([context] * cfg.n_fm)
        _, *fm = pipe.generate(sample_key, decode_key, pipe.params, st, contexts, jnp.asarray(designated))

        decisions = [fm, random_choice(rng, st, cfg.n_random), perturb(rng, st, *fm, cfg.n_perturb)]
        choice, power, mcs = (np.concatenate(a) for a in zip(*decisions))
        source = np.concatenate([
            np.where(designated[:, 0, 0] >= 0, SOURCES['fm_designated'], SOURCES['fm']),
            np.full(cfg.n_random, SOURCES['random']), np.full(cfg.n_perturb, SOURCES['perturbed'])
        ])
        thr, link_rate = create_simulator(scenario, st)(eval_key, choice, power, mcs)
        groups.append({
            'st': st, 'choice': choice, 'power': power, 'mcs': mcs, 'link_rate': link_rate.astype(np.float32),
            'thr': thr.astype(np.float32), 'source': source
        })

    return groups


def load_groups(cfg, data_path, paths, n_networks, seed):
    if not os.path.exists(path(data_path)):
        os.makedirs(os.path.dirname(path(data_path)), exist_ok=True)
        save_dataset(generate(cfg, paths, n_networks, seed), path(data_path))

    return load_dataset(path(data_path))


def group_batch(group):
    """
    Explicit graphs padded to power-of-two sizes, normalised link-rate targets and link masks of a group.
    """

    st = group['st']
    selected, tx_power, mcs = choice_to_arrays(st, *map(jnp.asarray, (group['choice'], group['power'], group['mcs'])))
    n_node_pad, n_edge_pad = nearest_power_of_two(st.is_ap.shape[0] + 1), nearest_power_of_two(st.senders.shape[0])
    graphs = pad_graph(explicit_graph(st, selected, tx_power, mcs), n_node_pad, n_edge_pad, xp=np)
    pad = ((0, 0), (0, n_edge_pad - st.link_ap.shape[0]))
    targets = np.pad(group['link_rate'] / LINK_SCALE, pad).astype(np.float32)
    mask = np.pad(np.asarray(selected), pad).astype(np.float32)
    return graphs, targets, mask, np.zeros(len(mask), dtype=np.int32)


@hydra.main(config_path='configs', config_name='surrogate', version_base=None)
def main(cfg):
    data = load_groups(cfg.data, cfg.data.train_path, cfg.data.train_paths, cfg.data.n_networks, cfg.data.seed)
    val_data = load_groups(cfg.data, cfg.data.val_path, cfg.data.val_paths, cfg.data.n_val_networks, cfg.data.seed)
    model = Surrogate(**cfg.model)

    def loss_fn(params, key, batch):
        loss, (nll, rank, _, _) = surrogate_loss(model, params, *batch, cfg.train.rank_weight, key)
        return loss, {'nll': nll, 'rank': rank}

    eval_fn = jax.jit(lambda params, batch: surrogate_loss(model, params, *batch, cfg.train.rank_weight))

    def val_fn(params, key):
        metrics = []

        for group in val_data:
            _, (nll, rank, pred, true) = eval_fn(params, group_batch(group))
            pred, true, fm = np.asarray(pred)[None], np.asarray(true)[None], group['source'] <= SOURCES['fm_designated']
            metrics.append([nll, rank, rank_correlation(pred, true), rank_correlation(pred[:, fm], true[:, fm])])

        return dict(zip(['nll', 'rank', 'spearman', 'spearman_fm'], np.nanmean(np.asarray(metrics, float), axis=0)))

    rng = np.random.default_rng(cfg.train.seed)
    order = np.concatenate([rng.permutation(len(data)) for _ in range(cfg.train.n_steps // len(data) + 1)])
    init_key, key = jax.random.split(jax.random.PRNGKey(cfg.train.seed))
    st = data[0]['st']
    arrays = choice_to_arrays(st, *(jnp.asarray(data[0][k][0]) for k in ('choice', 'power', 'mcs')))
    params = model.init({'params': init_key, 'dropout': init_key}, explicit_graph(st, *arrays), training=False)
    optimizer = create_optimizer(
        one_cycle(cfg.optimizer.learning_rate, cfg.train.n_steps), cfg.optimizer.weight_decay, cfg.optimizer.b1,
        cfg.optimizer.b2, cfg.optimizer.grad_clip
    )
    run_dir = path(os.path.join('runs', cfg.name))
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    fit(
        loss_fn, params, optimizer, (group_batch(data[i]) for i in order), cfg.train.n_steps, key, run_dir, val_fn,
        cfg.train.val_freq, cfg.train.log_freq, cfg.train.ckpt_freq
    )


if __name__ == '__main__':
    main()
