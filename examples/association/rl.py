import os
os.environ['JAX_ENABLE_X64'] = 'True'

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from examples.association.env import TRAIN_PARAMS, conflict_graph, evaluate, get_deployment, make_network
from examples.association.graph import build_topology, sta_targets
from examples.association.pipeline import (
    create_contexts,
    create_decoder,
    create_sampler,
    create_scorer,
    load_autoencoder,
    load_flow_matching,
    load_surrogate,
    plan_graphs,
    resolve
)
from fm4mac.checkpoints import Logger, create_checkpoint_manager, save_checkpoint
from fm4mac.flow import FlowMatching
from fm4mac.graphs import take_graphs
from fm4mac.metrics import device_metrics, pf_utility
from fm4mac.rl import create_nft_loss, group_advantages, local_utilities, nft_rewards, old_params_update
from fm4mac.search import rank_correlation
from fm4mac.surrogate import surrogate_loss
from fm4mac.training import create_optimizer, create_step_fn


def sta_neighbourhoods(deployment, net):
    """
    Relevant APs of each STA (its candidate APs and the APs they sense), shape [n_stas, n_aps]: the per-STA credit
    is the PF utility of the STAs associated with these APs.
    """

    conflict, _ = conflict_graph(np.asarray(deployment.ap_pos))
    reach = conflict | np.eye(len(conflict), dtype=bool)
    return (np.asarray(net.candidates).astype(float) @ reach > 0).astype(float)


@hydra.main(config_path='configs', config_name='rl', version_base=None)
def main(cfg):
    run_dir = resolve(os.path.join('runs', cfg.name))
    train = cfg.train
    fm_cfg, params_ref = load_flow_matching(cfg.flow_matching_path)
    fm = FlowMatching(**{**fm_cfg.model, 'dtype': train.nft.dtype})
    encode_fn, decode_fn = load_autoencoder(fm_cfg.autoencoder_path)
    surrogate, s_params = load_surrogate(cfg.surrogate_path)

    deployment = get_deployment(cfg.deployment)
    net, topology = make_network(deployment), build_topology(deployment)
    primary = jnp.asarray(deployment.primary)
    neighbourhoods = sta_neighbourhoods(deployment, net)
    fm_sample = FlowMatching(**{**fm_cfg.model, 'dtype': train.rollout.dtype})
    sample_fn = create_sampler(fm_sample, train.rollout.gen_steps, train.rollout.chunk_size)
    decode = create_decoder(decode_fn, topology, train.rollout.chunk_size)
    score = create_scorer(surrogate, topology, primary, train.rollout.chunk_size)

    def simulate(key, plans):
        plans, keys, size = jnp.asarray(plans), jax.random.split(key, len(plans)), train.rollout.sim_chunk_size
        return np.concatenate([
            np.asarray(evaluate(keys[i:i + size], net, plans[i:i + size], TRAIN_PARAMS))
            for i in range(0, len(plans), size)
        ])

    def rollout(key, params, contexts, n_sim_repeats):
        sample_key, plan_key, sim_key = jax.random.split(key, 3)
        z = sample_fn(params, sample_key, contexts)
        plans = np.asarray(decode(jax.random.split(plan_key, len(z)), contexts, z))
        thr = np.mean([simulate(key, plans) for key in jax.random.split(sim_key, n_sim_repeats)], axis=0)
        return z, plans, thr

    def evaluate_fm(key, params):
        n_samples, n_repeats = train.eval.n_samples, train.eval.n_repeats
        rollout_key, sim_key = jax.random.split(key)
        _, plans, thr = rollout(rollout_key, params, eval_contexts, 1)
        rewards = pf_utility(thr).reshape(-1, n_samples)
        scores = np.asarray(score(s_params, jnp.asarray(plans))).reshape(-1, n_samples)
        offsets = n_samples * np.arange(rewards.shape[0])[:, None]
        metrics = {
            'sample_mean': rewards.mean(), 'surrogate_rank_corr': rank_correlation(scores, rewards),
            'distinct_plans': np.mean([len(np.unique(p, axis=0)) for p in plans.reshape(*rewards.shape, -1)])
        }

        for top_k in train.eval.top_ks:
            idxs = (np.argsort(-scores, axis=-1)[:, :top_k] + offsets).reshape(-1)
            best = simulate(sim_key, np.repeat(plans[idxs], n_repeats, axis=0))
            best = best.reshape(-1, top_k, n_repeats, topology.n_stas).mean(2)
            metrics[f'top_{top_k}'] = pf_utility(best).mean()

            if top_k == 1:
                metrics |= {f'top_1_{k}': v for k, v in device_metrics(best[:, 0]).items() if k != 'pf'}

        return metrics

    nft_loss = create_nft_loss(fm, train.nft.beta, train.nft.adv_clip, train.nft.kl_coef)
    optimizer = create_optimizer(**cfg.optimizer, accumulation=train.nft.accumulation)
    nft_step = create_step_fn(lambda p, key, batch: nft_loss(p, key, *batch), optimizer)

    def s_loss(s_params, key, batch):
        plans, thr, groups = batch
        graphs, mask = plan_graphs(topology, primary, plans)
        return surrogate_loss(surrogate, s_params, graphs, sta_targets(topology, thr, jnp), mask, groups, 1.)

    s_optimizer = create_optimizer(train.surrogate.learning_rate, grad_clip=cfg.optimizer.grad_clip)
    s_step = create_step_fn(s_loss, s_optimizer)

    ctx_key, train_key = jax.random.split(jax.random.PRNGKey(train.seed))
    contexts_fn = lambda key, n: create_contexts(key, net, topology, primary, encode_fn, n)
    eval_contexts = contexts_fn(jax.random.PRNGKey(train.eval.seed), train.eval.n_steps)
    eval_contexts = take_graphs(eval_contexts, np.repeat(np.arange(train.eval.n_steps), train.eval.n_samples))
    contexts = contexts_fn(jax.random.split(ctx_key)[1], train.rollout.n_contexts)

    params, params_old = params_ref, params_ref
    opt_state, s_opt_state = optimizer.init(params), s_optimizer.init(s_params)
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    manager, logger = create_checkpoint_manager(run_dir), Logger(run_dir)
    n_groups, group_size = train.rollout.n_groups, train.rollout.group_size
    minibatch_size = train.nft.minibatch_size // train.nft.accumulation
    cand = topology.n_aps + np.arange(topology.n_candidates)

    for step in range(train.n_steps + 1):
        log = {}

        if step % train.eval.freq == 0 or step == train.n_steps:
            log |= {f'eval/{k}': v for k, v in evaluate_fm(jax.random.PRNGKey(train.eval.seed), params).items()}

        if step < train.n_steps:
            train_key, ctx_key, rollout_key = jax.random.split(train_key, 3)
            idxs = jax.random.choice(ctx_key, contexts.n_node.shape[0], shape=(n_groups,))
            batch_contexts = take_graphs(contexts, jnp.repeat(idxs, group_size))
            z, plans, thr = rollout(rollout_key, params_old, batch_contexts, train.rollout.n_sim_repeats)
            value = pf_utility(thr)
            rewards = nft_rewards(group_advantages(value, group_size), train.nft.adv_clip)
            rewards = np.repeat(rewards[:, None], z.shape[1], axis=1)

            # Per-STA credit: both directions of the candidate edges of a STA (its association decision) are
            # rewarded with the PF utility of the STAs associated with its relevant APs, group-relative per STA
            log_thr = np.log(np.maximum(thr, 1.))
            ap_values = np.stack([np.bincount(p, u, topology.n_aps) for p, u in zip(plans, log_thr)])
            local = group_advantages(local_utilities(ap_values, neighbourhoods), group_size)
            local = nft_rewards(local, train.nft.adv_clip)[:, topology.cand_sta]
            rewards[:, cand] = rewards[:, cand + topology.n_candidates] = local

            minibatches = np.split(np.arange(len(rewards)), range(minibatch_size, len(rewards), minibatch_size))
            losses, kls, s_losses = [], [], []
            train_key, perm_key = jax.random.split(train_key)

            for i in jax.random.permutation(perm_key, len(minibatches)).tolist():
                train_key, step_key = jax.random.split(train_key)
                mb = minibatches[i]
                batch = (take_graphs(batch_contexts, mb), z[mb], rewards[mb], params_old, params_ref)
                params, opt_state, loss, kl = nft_step(params, opt_state, step_key, batch)
                losses.append(loss)
                kls.append(kl)

            for _ in range(train.surrogate.n_epochs):
                train_key, perm_key = jax.random.split(train_key)

                for i in jax.random.permutation(perm_key, len(minibatches)).tolist():
                    mb = minibatches[i]
                    batch = (jnp.asarray(plans[mb]), jnp.asarray(thr[mb]), jnp.asarray(mb // group_size))
                    s_params, s_opt_state, loss, _ = s_step(s_params, s_opt_state, None, batch)
                    s_losses.append(loss)

            params_old, eta = old_params_update(params, params_old, step, train.nft.eta_rate, train.nft.eta_max)
            log |= {
                'train/reward_mean': value.mean(), 'train/reward_best': value.reshape(-1, group_size).max(-1).mean(),
                'rl/loss': np.mean(losses), 'rl/kl': np.mean(kls), 'rl/eta': eta, 'surrogate/loss': np.mean(s_losses)
            }

        logger(step, log)

        if step % train.ckpt_freq == 0 or step == train.n_steps:
            save_checkpoint(manager, step, params=params, surrogate_params=s_params)

    manager.wait_until_finished()
    manager.close()
    logger.close()


if __name__ == '__main__':
    main()
