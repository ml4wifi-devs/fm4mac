import flax.linen as nn
import jax.numpy as jnp
import jraph
from jraph._src.utils import segment_mean

from fm4mac.graphs import segment_ids


class MLP(nn.Module):
    features: tuple
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)

        for size in self.features:
            x = nn.Dense(size, dtype=self.dtype)(x)
            x = nn.gelu(x)

        return x + nn.Dense(self.features[-1], dtype=self.dtype)(residual)


def make_embed_fn(features, dtype):
    @jraph.concatenated_args
    def embed_fn(inputs):
        return nn.Dense(features, dtype=dtype)(inputs)

    return embed_fn


def make_mlp(features, dtype):
    @jraph.concatenated_args
    def update_fn(inputs):
        return MLP(features, dtype)(inputs)

    return update_fn


class FeedForwardBlock(nn.Module):
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training=True):
        out_dim = x.shape[-1]
        x = nn.Dense(self.ff_dim, dtype=self.dtype, use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(out_dim, dtype=self.dtype, use_bias=False)(x)
        return nn.Dropout(self.drop_rate)(x, deterministic=not training)


class TransformerBlock(nn.Module):
    n_heads: int
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask, training=True):
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=x.shape[-1], dtype=self.dtype, use_bias=False
        )(x, mask=mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = FeedForwardBlock(self.ff_dim, self.drop_rate, self.dtype)(x, training=training)
        return x + residual


class Transformer(nn.Module):
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        x = nn.Dense(self.dim, dtype=self.dtype, use_bias=False)(x)

        for _ in range(self.n_layers):
            x = TransformerBlock(self.n_heads, self.ff_dim, self.drop_rate, self.dtype)(x, mask=mask, training=training)

        return nn.LayerNorm(dtype=self.dtype)(x)


def make_transformer(dim, ff_dim, n_layers, n_heads, drop_rate, dtype, ids=None, training=True):
    """
    Graph network update function: a transformer over all elements of a batch with attention masked to elements
    of the same graph (`ids`).
    """

    mask = None if ids is None else ids[:, None] == ids[None, :]

    @jraph.concatenated_args
    def update_fn(inputs):
        return Transformer(dim, ff_dim, n_layers, n_heads, drop_rate, dtype)(inputs, mask=mask, training=training)

    return update_fn


def transformer_graph_network(x, dim, ff_dim, n_layers, n_heads, drop_rate, dtype, training):
    """
    Graph network layer with segment-masked transformers as the edge, node and global update functions.
    """

    transformer_fn = lambda ids: make_transformer(dim, ff_dim, n_layers, n_heads, drop_rate, dtype, ids, training)
    return jraph.GraphNetwork(
        update_edge_fn=transformer_fn(segment_ids(x.n_edge, x.senders.shape[0])),
        update_node_fn=transformer_fn(segment_ids(x.n_node, x.nodes.shape[0])),
        update_global_fn=transformer_fn(jnp.arange(x.n_node.shape[0])),
        aggregate_edges_for_nodes_fn=segment_mean,
        aggregate_nodes_for_globals_fn=segment_mean,
        aggregate_edges_for_globals_fn=segment_mean
    )(x)


def mlp_graph_network(x, features, dtype):
    return jraph.GraphNetwork(
        update_edge_fn=make_mlp(features, dtype),
        update_node_fn=make_mlp(features, dtype),
        update_global_fn=make_mlp(features, dtype),
        aggregate_edges_for_nodes_fn=segment_mean,
        aggregate_nodes_for_globals_fn=segment_mean,
        aggregate_edges_for_globals_fn=segment_mean
    )(x)
