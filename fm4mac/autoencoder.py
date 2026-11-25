from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import optax

from fm4mac.graphs import clear_graph, head_sizes
from fm4mac.nn import make_embed_fn, mlp_graph_network


class Encoder(nn.Module):
    embed_features: int
    mlp_features: Sequence[int]
    latent_features: int
    n_layers: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, graph):
        x = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.embed_features, self.dtype))(graph)

        for _ in range(self.n_layers):
            x = mlp_graph_network(x, self.mlp_features, self.dtype)

        z_mean = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.latent_features, self.dtype))(x)
        z_log_var = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.latent_features, self.dtype))(x)
        return z_mean, z_log_var


class Decoder(nn.Module):
    """
    Decodes edge latents into the logits (categorical attributes) or values (scalar attributes) of `edge_cls`.
    """

    edge_cls: type
    mlp_features: Sequence[int]
    n_layers: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, z):
        for _ in range(self.n_layers):
            z = mlp_graph_network(z, self.mlp_features, self.dtype)

        x = nn.gelu(z.edges)
        heads = {name: nn.Dense(size, dtype=self.dtype)(x) for name, size in head_sizes(self.edge_cls).items()}
        return z._replace(edges=self.edge_cls(**heads))


class Autoencoder(nn.Module):
    """
    Variational GNN autoencoder mapping every edge of a network graph (topology, measurements and decisions)
    to a latent vector.
    """

    edge_cls: type
    embed_features: int
    encoder_features: Sequence[int]
    decoder_features: Sequence[int]
    latent_features: int
    n_layers: int
    dtype: jnp.dtype

    def setup(self):
        self.encoder = Encoder(
            self.embed_features, self.encoder_features, self.latent_features, self.n_layers, self.dtype
        )
        self.decoder = Decoder(self.edge_cls, self.decoder_features, self.n_layers, self.dtype)

    def __call__(self, graph):
        z_mean, z_log_var = self.encoder(graph)
        epsilon = jax.random.normal(self.make_rng('sample'), z_mean.edges.shape, dtype=z_mean.edges.dtype)
        z = clear_graph(z_mean._replace(edges=z_mean.edges + jnp.exp(0.5 * z_log_var.edges) * epsilon))
        return self.decoder(z), z_mean, z_log_var

    def encode(self, graph):
        return self.encoder(graph)[0]

    def decode(self, z):
        return self.decoder(clear_graph(z))


def mixture_categorical_loss(logits, labels):
    """
    Binary cross-entropy of the `NA` class plus categorical cross-entropy of the other classes when applicable.
    """

    na_loss = optax.sigmoid_binary_cross_entropy(logits[..., -1], labels[..., -1])
    softmax_loss = optax.softmax_cross_entropy(logits[..., :-1], labels[..., :-1])
    return na_loss + (1 - labels[..., -1]) * softmax_loss


def mixture_accuracy(logits, labels):
    """
    Accuracy of the `NA` class, accuracy of the other classes where applicable, and the applicability mask.
    """

    na = labels[..., -1]
    na_acc = (logits[..., -1] > 0.) == na
    acc = (1 - na) * (logits[..., :-1].argmax(axis=-1) == labels[..., :-1].argmax(axis=-1))
    return jnp.asarray([na_acc, acc, 1 - na])


def kl_loss(mean, log_var):
    return -0.5 * (1. + log_var - mean ** 2 - jnp.exp(log_var)).mean(axis=-1)


def autoencoder_loss(pred, batch, z_mean, z_log_var, field_losses, weights, kl_weight):
    """
    Weighted sum of per-attribute losses (`field_losses`: name -> loss(pred, target) per edge, padding excluded)
    and the KL term. Returns the loss, the KL term and the per-attribute losses.
    """

    mask = jraph.get_edge_padding_mask(pred)
    n = mask.sum()
    losses = jnp.asarray([
        (mask * fn(getattr(pred.edges, name), getattr(batch.edges, name))).sum() / n
        for name, fn in field_losses.items()
    ])
    kl = (mask * kl_loss(z_mean.edges, z_log_var.edges)).sum() / n
    return losses @ jnp.asarray(weights) + kl_weight * kl, kl, losses


def squared_error(pred, target):
    return optax.squared_error(pred, target).mean(axis=-1)


def binary_cross_entropy(pred, target):
    return optax.sigmoid_binary_cross_entropy(pred, target).mean(axis=-1)
