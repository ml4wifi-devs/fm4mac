import jax
import jax.numpy as jnp
import numpy as np
import optax

from fm4mac.flow import append_noise


def create_nft_loss(model, beta, adv_clip, kl_coef):
    """
    DiffusionNFT (https://arxiv.org/abs/2509.16117) for flow matching on edge latents. `reward` in [0, 1] is either one
    value per sample or one per edge, so credit can be assigned per device. The update moves the model towards
    samples with reward above 0.5 (better than their group) and away from the others, with a KL penalty towards
    the pretrained (reference) model.
    """

    def x0_loss(v, x_t, x, t):
        err = x_t + (1 - t) * v - x
        return (err ** 2).mean(axis=-1) / jax.lax.stop_gradient(jnp.maximum(jnp.abs(err).mean(), 1e-5))

    def nft_loss(params, key, context, x, reward, params_old, params_ref, cond=None):
        t_key, edges_key, nodes_key, globals_key = jax.random.split(key, 4)
        t = jnp.full((x.shape[0], 1), jax.random.uniform(t_key, maxval=0.99, dtype=x.dtype))
        x_t = (1 - t) * jax.random.normal(edges_key, x.shape, dtype=x.dtype) + t * x
        cx = context._replace(
            edges=jnp.concatenate([context.edges, x_t[:, None]], axis=1),
            nodes=append_noise(nodes_key, context.nodes),
            globals=append_noise(globals_key, context.globals)
        )

        v_pred, v_old, v_ref = (
            model.apply(p, cx, t, training=False, cond=cond).edges for p in (params, params_old, params_ref)
        )
        v_plus = (1 - beta) * v_old + beta * v_pred
        v_minus = (1 + beta) * v_old - beta * v_pred
        loss = reward * x0_loss(v_plus, x_t, x, t) + (1 - reward) * x0_loss(v_minus, x_t, x, t)
        kl = ((v_pred - v_ref) ** 2).mean()
        return loss.mean() * adv_clip / beta + kl_coef * kl, kl

    def loss_fn(params, key, contexts, x, rewards, params_old, params_ref, cond=None):
        """
        Mean NFT loss of a batch of samples (contexts, latents `x`, rewards and conditions stacked on axis 0).
        """

        keys = jax.random.split(key, rewards.shape[0])
        loss, kl = jax.vmap(nft_loss, in_axes=(None, 0, 0, 0, 0, None, None, 0))(
            params, keys, contexts, x, rewards, params_old, params_ref, cond
        )
        return loss.mean(), kl.mean()

    return loss_fn


def group_advantages(values, group_size, scale=None):
    """
    Advantages relative to the mean of each group of samples generated for the same observation, normalised by the
    standard deviation over all samples (or `scale`). `values` is `[n, ...]` with consecutive groups on axis 0.
    """

    groups = values.reshape((-1, group_size) + values.shape[1:])
    adv = groups - groups.mean(axis=1, keepdims=True)
    scale = values.std(axis=0) if scale is None else scale
    return (adv / (scale + 1e-4)).reshape(values.shape)


def nft_rewards(advantages, adv_clip):
    return 0.5 + 0.5 * np.clip(advantages / adv_clip, -1., 1.)


def local_utilities(device_values, neighbourhoods):
    """
    Credit assignment for per-device decisions: the utility of each decision's neighbourhood instead of the whole
    network, which says little about any single decision in a large network. `device_values` is `[..., n_devices]`
    and `neighbourhoods` a `[n_decisions, n_devices]` 0/1 relevance matrix (e.g. per AP including the AP itself, or
    per station its candidate APs and their interferers).
    """

    return device_values @ neighbourhoods.T


def old_params_update(params, params_old, step, eta_rate, eta_max):
    """
    Soft update of the sampling (old) model with a decay growing linearly from 0 to `eta_max`.
    """

    eta = min(eta_rate * step, eta_max)
    return optax.incremental_update(params, params_old, step_size=1 - eta), eta
