import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax

from fm4mac.graphs import SHARED_STRUCTURE
from fm4mac.nn import make_embed_fn, transformer_graph_network


class Surrogate(nn.Module):
    """
    Surrogate on explicit configuration graphs: input embeddings, residual graph network layers with
    segment-masked transformers, and a per-edge Gaussian head (mean, scale) of a per-device performance metric.
    The predicted objective of a configuration is the sum of the means over its target edges, so prediction errors
    stay local and the model transfers to networks larger than those seen in training.
    """

    gnn_layers: int
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float
    dtype: jnp.dtype
    embed_nodes: bool = True

    @nn.compact
    def __call__(self, x, training=True):
        x = jraph.GraphMapFeatures(
            embed_edge_fn=make_embed_fn(self.dim, self.dtype),
            embed_node_fn=make_embed_fn(self.dim, self.dtype) if self.embed_nodes else None
        )(x)

        for i in range(self.gnn_layers):
            x_new = transformer_graph_network(
                x, self.dim, self.ff_dim, self.n_layers, self.n_heads, self.drop_rate, self.dtype, training
            )
            x = x_new if i == 0 else x_new._replace(
                nodes=x_new.nodes + x.nodes, edges=x_new.edges + x.edges, globals=x_new.globals + x.globals
            )

        out = nn.Dense(2, dtype=self.dtype)(nn.gelu(x.edges))
        return out[..., 0], nn.softplus(out[..., 1]) + 1e-3


def apply_batch(model, params, graphs, key=None):
    """
    Applies the surrogate to graphs sharing one structure (features stacked on the first axis); with dropout when a
    key is given.
    """

    if key is None:
        return jax.vmap(lambda g: model.apply(params, g, training=False), in_axes=(SHARED_STRUCTURE,))(graphs)

    keys = jax.random.split(key, graphs.edges.shape[0])
    return jax.vmap(
        lambda g, k: model.apply(params, g, training=True, rngs={'dropout': k}), in_axes=(SHARED_STRUCTURE, 0)
    )(graphs, keys)


def utility(values, mask):
    """
    Objective of each configuration: the sum of per-device values over its target edges (`mask`, aligned with the
    first edges of the graph).
    """

    return (values[..., :mask.shape[-1]] * mask).sum(axis=-1)


def rank_loss(pred, true, groups):
    """
    Pairwise logistic ranking loss between configurations of the same group (network and observation).
    """

    pairs = (groups[:, None] == groups[None]) & (true[:, None] != true[None])
    labels = (true[:, None] > true[None]).astype(pred.dtype)
    loss = optax.sigmoid_binary_cross_entropy(pred[:, None] - pred[None], labels)
    return (pairs * loss).sum() / jnp.maximum(pairs.sum(), 1)


def surrogate_loss(model, params, graphs, targets, mask, groups, rank_weight, key=None):
    """
    Gaussian NLL of the per-device targets on the target edges plus the within-group ranking loss of the
    predicted vs. measured objective. Selecting candidates needs a correct ranking more than accurate values.
    """

    mean, scale = apply_batch(model, params, graphs, key)
    mean, scale = mean[:, :mask.shape[-1]], scale[:, :mask.shape[-1]]
    nll = 0.5 * ((targets - mean) / scale) ** 2 + jnp.log(scale)
    nll = (mask * nll).sum() / jnp.maximum(mask.sum(), 1)

    pred, true = utility(mean, mask), utility(targets, mask)
    rank = rank_loss(pred, true, groups)
    return nll + rank_weight * rank, (nll, rank, pred, true)
