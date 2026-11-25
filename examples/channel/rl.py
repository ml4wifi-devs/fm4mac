import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from examples.channel.env import decode_action, evaluate, get_deployment, objective
from examples.channel.graph import build_topology, neighbourhoods, plan_graph
from examples.channel.pipeline import (
    ROOT,
    create_sampler,
    create_scorer,
    load_autoencoder,
    load_flow_matching,
    load_surrogate,
    observe
)
from fm4mac.checkpoints import Logger, create_checkpoint_manager, save_checkpoint
from fm4mac.graphs import take_graphs
from fm4mac.metrics import device_metrics
from fm4mac.rl import create_nft_loss, group_advantages, local_utilities, nft_rewards, old_params_update
from fm4mac.search import rank_correlation
from fm4mac.surrogate import surrogate_loss
from fm4mac.training import create_optimizer, create_step_fn


@hydra.main(config_path='configs', config_name='rl', version_base=None)
def main(cfg):
    fm, params_ref, autoencoder_path = load_flow_matching(cfg.flow_matching_path)
    encode_fn, decode_fn = load_autoencoder(autoencoder_path)
    surrogate, s_params = load_surrogate(cfg.surrogate_path)

    nft = cfg.nft
    nft_loss = create_nft_loss(fm, nft.beta, nft.adv_clip, nft.kl_coef)
    optimizer = create_optimizer(nft.learning_rate, accumulation=nft.accumulation)
    step_fn = create_step_fn(lambda p, k, batch: nft_loss(p, k, *batch), optimizer)
    s_optimizer = create_optimizer(cfg.surrogate.learning_rate)

    def simulate(key, env, plans):
        plans = np.asarray(plans)
        keys = jax.random.split(key, len(plans))
        chunk = cfg.rollout.sim_chunk_size
        return np.concatenate([
            np.asarray(evaluate(keys[i:i + chunk], *env['deployment'], plans[i:i + chunk]))
            for i in range(0, len(plans), chunk)
        ])

    def rollout(key, params, contexts, env):
        sample_key, plan_key, sim_key = jax.random.split(key, 3)
        z, plans = env['sample'](params, sample_key, plan_key, contexts)
        # the split keeps the simulation seeds of the tutorial runs
        thr = simulate(jax.random.split(sim_key, 1)[0], env, plans)
        return z, np.asarray(plans), thr

    def evaluate_env(key, params, s_params, env):
        n_samples, n_repeats = cfg.eval.n_samples, cfg.eval.n_repeats
        rollout_key, sim_key = jax.random.split(key)
        _, plans, thr = rollout(rollout_key, params, env['eval_contexts'], env)
        values = np.asarray(objective(thr)).reshape(-1, n_samples)
        scores = np.asarray(env['score'](s_params, jnp.asarray(plans))).reshape(-1, n_samples)
        metrics = {
            'sample_mean': values.mean(),
            'distinct_plans': np.mean([len(np.unique(p, axis=0)) for p in plans.reshape(*values.shape, -1)]),
            'surrogate_rank_corr': rank_correlation(scores, values)
        }

        for top_k in cfg.eval.top_ks:
            idxs = (np.argsort(-scores, axis=-1)[:, :top_k] + n_samples * np.arange(len(values))[:, None]).ravel()
            thr = simulate(sim_key, env, np.repeat(plans[idxs], n_repeats, axis=0))
            metrics[f'top_{top_k}'] = np.asarray(objective(thr)).mean()

            if top_k == 1:
                top_1 = device_metrics(thr.reshape(-1, n_repeats, env['n_aps']).mean(1))
                metrics |= {f'top_1_{k}': v for k, v in top_1.items() if k != 'pf'}

        return metrics

    def create_env(name, ctx_key):
        deployment = get_deployment(name)
        topology = build_topology(*deployment)
        n_aps = topology.n_aps

        def s_loss_fn(s_params, _, batch):
            plans, thr, groups = batch
            graphs = plan_graph(topology, *decode_action(plans), xp=jnp)
            targets = jnp.log(jnp.maximum(thr, 1.)).astype(jnp.float32)
            return surrogate_loss(surrogate, s_params, graphs, targets, jnp.ones_like(targets), groups, 1.)

        env = dict(
            name=name, deployment=deployment, n_aps=n_aps, neighbours=neighbourhoods(topology),
            sample=create_sampler(fm, decode_fn, n_aps, cfg.rollout.chunk_size, cfg.rollout.gen_steps),
            score=create_scorer(surrogate, topology, cfg.rollout.chunk_size),
            s_step=create_step_fn(s_loss_fn, s_optimizer)
        )
        eval_contexts = observe(jax.random.PRNGKey(cfg.eval.seed), deployment, topology, encode_fn, cfg.eval.n_contexts)
        env['eval_contexts'] = take_graphs(eval_contexts, np.repeat(np.arange(cfg.eval.n_contexts), cfg.eval.n_samples))
        env['contexts'] = observe(ctx_key, deployment, topology, encode_fn, cfg.rollout.n_contexts)
        return env

    key = jax.random.PRNGKey(cfg.seed)
    ctx_key, train_key = jax.random.split(key)
    envs = []

    for name in cfg.deployments:
        ctx_key, pool_key = jax.random.split(ctx_key)
        envs.append(create_env(name, pool_key))

    params, params_old = params_ref, params_ref
    opt_state, s_opt_state = optimizer.init(params), s_optimizer.init(s_params)

    run_dir = os.path.join(ROOT, 'runs', cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, 'config.yaml'))
    manager, logger = create_checkpoint_manager(run_dir), Logger(run_dir)

    n_groups = cfg.rollout.n_groups // len(envs)
    group_size = cfg.rollout.group_size
    minibatch_size = nft.minibatch_size // nft.accumulation
    start_time = time.time()

    for step in range(cfg.n_steps + 1):
        log = {'time': time.time() - start_time}

        if step % cfg.eval.freq == 0 or step == cfg.n_steps:
            for env in envs:
                metrics = evaluate_env(jax.random.PRNGKey(cfg.eval.seed), params, s_params, env)
                log |= {f'eval/{env["name"]}/{k}': v for k, v in metrics.items()}

        if step < cfg.n_steps:
            minibatches = []

            for env in envs:
                train_key, ctx_key, rollout_key = jax.random.split(train_key, 3)
                idxs = jax.random.choice(ctx_key, env['contexts'].n_node.shape[0], shape=(n_groups,))
                contexts = take_graphs(env['contexts'], jnp.repeat(idxs, group_size))
                z, plans, thr = rollout(rollout_key, params_old, contexts, env)
                values = np.asarray(objective(thr))

                rewards = nft_rewards(group_advantages(values, group_size), nft.adv_clip)
                rewards = np.repeat(rewards[:, None], z.shape[1], axis=1)
                local = local_utilities(np.log(np.maximum(thr, 1.)), env['neighbours'])
                rewards[:, :env['n_aps']] = nft_rewards(group_advantages(local, group_size), nft.adv_clip)

                for i in range(0, len(rewards), minibatch_size):
                    idxs = np.arange(i, min(i + minibatch_size, len(rewards)))
                    minibatches.append((
                        take_graphs(contexts, idxs), z[idxs], rewards[idxs],
                        plans[idxs], thr[idxs], idxs // group_size, env
                    ))

                groups = values.reshape(n_groups, group_size)
                log[f'train/{env["name"]}/reward_mean'] = values.mean()
                log[f'train/{env["name"]}/reward_best'] = groups.max(axis=-1).mean()

            losses, kls, s_losses = [], [], []

            for _ in range(nft.n_epochs):
                train_key, perm_key = jax.random.split(train_key)

                for i in jax.random.permutation(perm_key, len(minibatches)).tolist():
                    train_key, step_key = jax.random.split(train_key)
                    contexts, z, rewards, *_ = minibatches[i]
                    params, opt_state, loss, kl = step_fn(
                        params, opt_state, step_key, (contexts, z, rewards, params_old, params_ref)
                    )
                    losses.append(loss)
                    kls.append(kl)

            for _ in range(cfg.surrogate.n_epochs):
                train_key, perm_key = jax.random.split(train_key)

                for i in jax.random.permutation(perm_key, len(minibatches)).tolist():
                    *_, plans, thr, groups, env = minibatches[i]
                    s_params, s_opt_state, s_loss, _ = env['s_step'](s_params, s_opt_state, None, (plans, thr, groups))
                    s_losses.append(s_loss)

            params_old, eta = old_params_update(params, params_old, step, nft.eta_rate, nft.eta_max)
            log |= {'rl/loss': np.mean(losses), 'rl/kl': np.mean(kls), 'rl/eta': eta}
            log |= {'surrogate/loss': np.mean(s_losses)} if s_losses else {}

        logger(step, log)

        if step % cfg.ckpt_freq == 0 or step == cfg.n_steps:
            save_checkpoint(manager, step, params=params, surrogate_params=s_params)

    manager.wait_until_finished()
    manager.close()
    logger.close()


if __name__ == '__main__':
    main()
