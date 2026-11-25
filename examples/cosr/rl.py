import os
os.environ['JAX_ENABLE_X64'] = 'True'

import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from examples.cosr.env import get_scenario, sample_designated
from examples.cosr.graph import designated_edges
from examples.cosr.pipeline import Pipeline, create_nft_step, create_surrogate_step, path, surrogate_config
from examples.cosr.evaluate import create_env, evaluate, rollout
from fm4mac.checkpoints import Logger, create_checkpoint_manager, load_config, save_checkpoint
from fm4mac.graphs import take_graphs
from fm4mac.rl import group_advantages, nft_rewards, old_params_update
from fm4mac.training import create_optimizer


@hydra.main(config_path='configs', config_name='rl', version_base=None)
def main(cfg):
    """
    Online mode: DiffusionNFT fine-tuning of the pretrained FM on simulated rollouts in one deployment, with the
    explicit surrogate retrained on the same rollouts and used for selection and refinement in evaluation.
    """

    train, nft = cfg.train, cfg.train.nft
    pipe = Pipeline(
        cfg.flow_matching_path, cfg.surrogate_path, train.rollout.gen_steps, train.rollout.chunk_size,
        refine_batch_size=cfg.eval.refine_batch_size
    )
    scenario = get_scenario(cfg.scenario)
    ctx_key, train_key = jax.random.split(jax.random.PRNGKey(train.seed))
    _, pool_key = jax.random.split(ctx_key)
    env = create_env(pipe, scenario, train.n_designated, cfg.eval, train.rollout.sim_chunk_size)
    pool = pipe.contexts(pool_key, scenario, env['st'], train.rollout.n_contexts)
    params = params_old = params_ref = pipe.params
    s_params = pipe.s_params

    optimizer = create_optimizer(
        cfg.optimizer.learning_rate, cfg.optimizer.weight_decay, grad_clip=cfg.optimizer.grad_clip
    )
    s_optimizer = create_optimizer(train.surrogate.learning_rate, grad_clip=cfg.optimizer.grad_clip)
    opt_state, s_opt_state = optimizer.init(params), s_optimizer.init(s_params)
    nft_step = create_nft_step(pipe.fm, optimizer, nft.beta, nft.adv_clip, nft.kl_coef)
    surrogate_step = create_surrogate_step(pipe.surrogate, s_optimizer, train.surrogate.rank_weight)

    run_dir = path(os.path.join('runs', cfg.name))
    os.makedirs(run_dir, exist_ok=True)
    run_cfg = {'surrogate': {'model': surrogate_config(cfg.surrogate_path)[0]}, 'rl': cfg}
    run_cfg = OmegaConf.merge(load_config(path(cfg.flow_matching_path)), run_cfg)
    OmegaConf.save(run_cfg, os.path.join(run_dir, 'config.yaml'))
    manager, logger = create_checkpoint_manager(run_dir), Logger(run_dir)

    n_groups, group_size, minibatch_size = train.rollout.n_groups, train.rollout.group_size, nft.minibatch_size
    start = time.time()

    for step in range(train.n_steps + 1):
        log = {'time': time.time() - start}

        if step % cfg.eval.freq == 0 or step == train.n_steps:
            refine = cfg.eval.refine and (
                cfg.eval.refine_freq is None or step % cfg.eval.refine_freq == 0 or step == train.n_steps
            )
            metrics = evaluate(jax.random.PRNGKey(cfg.eval.seed), pipe, params, s_params, env, cfg.eval, refine)
            log |= {f'eval/{k}': v for k, v in metrics.items()}

        if step < train.n_steps:
            train_key, ctx_key, rollout_key, designated_key, nft_key, s_key = jax.random.split(train_key, 6)
            idxs = jnp.repeat(jax.random.choice(ctx_key, train.rollout.n_contexts, (n_groups,)), group_size)
            contexts = take_graphs(pool, idxs)
            designated = sample_designated(designated_key, scenario.associations, train.n_designated, n_groups)
            designated = jnp.repeat(designated, group_size, axis=0)
            cond = jax.vmap(designated_edges)(contexts, designated)

            z, decisions, rewards_raw, link_rates = rollout(
                pipe, rollout_key, params_old, env, contexts, designated, train.rollout.n_sim_repeats
            )
            rewards = nft_rewards(group_advantages(rewards_raw, group_size), nft.adv_clip)
            groups = np.arange(len(rewards)) // group_size
            batches = np.array_split(np.arange(len(rewards)), np.arange(minibatch_size, len(rewards), minibatch_size))
            losses, kls, s_losses = [], [], []

            for _ in range(nft.n_epochs):
                nft_key, perm_key = jax.random.split(nft_key)

                for i in jax.random.permutation(perm_key, len(batches)).tolist():
                    nft_key, step_key = jax.random.split(nft_key)
                    b = batches[i]
                    batch = (take_graphs(contexts, b), z[b], jnp.asarray(rewards[b]), params_old, params_ref, cond[b])
                    params, opt_state, loss, metrics = nft_step(params, opt_state, step_key, batch)
                    losses.append(loss)
                    kls.append(metrics['kl'])

            for _ in range(train.surrogate.n_epochs):
                s_key, perm_key = jax.random.split(s_key)

                for i in jax.random.permutation(perm_key, len(batches)).tolist():
                    b = batches[i]
                    batch = (env['st'], *(jnp.asarray(a[b]) for a in decisions), jnp.asarray(link_rates[b]), groups[b])
                    s_params, s_opt_state, s_loss, _ = surrogate_step(s_params, s_opt_state, None, batch)
                    s_losses.append(s_loss)

            params_old, eta = old_params_update(params, params_old, step, nft.eta_rate, nft.eta_max)
            groups_raw = rewards_raw.reshape(n_groups, group_size)
            log |= {
                'train/reward_mean': rewards_raw.mean(), 'train/reward_best': groups_raw.max(axis=-1).mean(),
                'train/reward_group_std': groups_raw.std(axis=-1).mean(), 'rl/loss': np.mean(losses),
                'rl/kl': np.mean(kls), 'rl/eta': eta, 'surrogate/loss': np.mean(s_losses)
            }

        logger(step, log)

        if step % train.ckpt_freq == 0 or step == train.n_steps:
            save_checkpoint(manager, step, ema_params=params, surrogate_params=s_params)

    manager.wait_until_finished()
    manager.close()
    logger.close()


if __name__ == '__main__':
    main()
