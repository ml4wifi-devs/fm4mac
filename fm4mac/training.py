import itertools

import jax
import optax

from fm4mac.checkpoints import Logger, create_checkpoint_manager, save_checkpoint


def create_optimizer(learning_rate, weight_decay=0., b1=0.9, b2=0.999, grad_clip=1., accumulation=1):
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay)
    )
    return optax.MultiSteps(optimizer, accumulation) if accumulation > 1 else optimizer


def one_cycle(peak_value, n_steps, pct_start=0.1):
    return optax.cosine_onecycle_schedule(n_steps, peak_value, pct_start)


def create_step_fn(loss_fn, optimizer):
    """
    Jitted optimisation step for `loss_fn(params, key, batch) -> (loss, metrics)`.
    """

    @jax.jit
    def step_fn(params, opt_state, key, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, metrics

    return step_fn


def fit(
        loss_fn, params, optimizer, batches, n_steps, key, run_dir,
        val_fn=None, val_freq=1000, log_freq=100, ckpt_freq=5000, ema_decay=None
):
    """
    Generic training loop: one step per batch of the (possibly infinite) iterable `batches`, validation metrics from
    `val_fn(params, key) -> dict` (on the EMA parameters if `ema_decay` is set), JSON-lines logs and orbax
    checkpoints of `params` (and `ema_params`) in `run_dir`.
    """

    step_fn = create_step_fn(loss_fn, optimizer)
    opt_state = optimizer.init(params)
    ema_params = params
    manager, logger = create_checkpoint_manager(run_dir), Logger(run_dir)
    last = n_steps - 1

    for step, batch in zip(range(n_steps), itertools.cycle(batches) if isinstance(batches, list) else batches):
        key, step_key, val_key = jax.random.split(key, 3)
        params, opt_state, loss, metrics = step_fn(params, opt_state, step_key, batch)

        if ema_decay is not None:
            ema_params = optax.incremental_update(params, ema_params, 1. - ema_decay)

        log = {}

        if step % log_freq == 0 or step == last:
            log |= {'train/loss': loss} | {f'train/{k}': v for k, v in metrics.items()}

        if val_fn is not None and (step % val_freq == 0 or step == last):
            log |= {f'val/{k}': v for k, v in val_fn(params if ema_decay is None else ema_params, val_key).items()}

        if log:
            logger(step, log)

        if step % ckpt_freq == 0 or step == last:
            items = {'params': params} | ({'ema_params': ema_params} if ema_decay is not None else {})
            save_checkpoint(manager, step, **items)

    manager.wait_until_finished()
    manager.close()
    logger.close()
    return params, ema_params
