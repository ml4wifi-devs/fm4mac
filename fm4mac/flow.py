import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph

from fm4mac.graphs import clear_graph
from fm4mac.nn import FeedForwardBlock, make_embed_fn, transformer_graph_network


class TimeEmbedding(nn.Module):
    ff_dim: int
    latent_size: int
    drop_rate: float
    dtype: jnp.dtype

    @staticmethod
    def sinusoidal(t, dim, min_timescale=1., max_timescale=1e4):
        n = float(dim // 2)
        log_increment = jnp.log(max_timescale / min_timescale) / n
        inv_timescales = min_timescale * jnp.exp(jnp.arange(n, dtype=jnp.float64) * -log_increment)
        t = t * inv_timescales[None, :]
        return jnp.concatenate([jnp.cos(t), jnp.sin(t)], axis=-1)

    @nn.compact
    def __call__(self, t, training=True):
        t = self.sinusoidal(t, self.latent_size)
        return FeedForwardBlock(self.ff_dim, self.drop_rate, self.dtype)(t, training=training)


class FlowMatching(nn.Module):
    """
    Conditional flow matching on edge latents. Inputs have shape `[n, n_context + 1, d]`: the latents of the
    observed context(s) followed by the noisy target. An optional per-edge condition (e.g. designated devices) is
    concatenated to the edge inputs.
    """

    gnn_layers: int
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float
    embed_features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, t, training=True, cond=None):
        x = x._replace(
            edges=x.edges.reshape(x.edges.shape[0], -1),
            nodes=x.nodes.reshape(x.nodes.shape[0], -1),
            globals=x.globals.reshape(x.globals.shape[0], -1)
        )
        t = TimeEmbedding(x.edges.shape[-1], x.edges.shape[-1], self.drop_rate, self.dtype)(t, training=training)
        x = x._replace(edges=x.edges + t)

        if cond is not None:
            x = x._replace(edges=jnp.concatenate([x.edges, cond.astype(x.edges.dtype)], axis=-1))

        for _ in range(self.gnn_layers):
            x = transformer_graph_network(
                x, self.dim, self.ff_dim, self.n_layers, self.n_heads, self.drop_rate, self.dtype, training
            )

        return jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.embed_features, self.dtype))(x)

    def gen(self, context, n_steps=6, cond=None):
        """
        Samples target latents with `n_steps` Euler steps from Gaussian noise.
        """

        def scan_fn(model, x, t):
            cx = context._replace(
                edges=jnp.concatenate([context.edges, x.edges[:, None]], axis=1),
                nodes=jnp.concatenate([context.nodes, x.nodes[:, None]], axis=1),
                globals=jnp.concatenate([context.globals, x.globals[:, None]], axis=1)
            )
            v = model(cx, t, training=False, cond=cond)
            return x._replace(edges=x.edges + v.edges / n_steps), None

        keys = dict(zip(('edges', 'nodes', 'globals'), jax.random.split(self.make_rng('sample'), 3)))
        shape = lambda x: (x.shape[0], x.shape[-1])
        z = context._replace(**{
            name: jax.random.normal(key, shape(getattr(context, name)), self.dtype) for name, key in keys.items()
        })
        scan = nn.scan(scan_fn, variable_broadcast='params')
        x, _ = scan(self, z, jnp.linspace(0., 1., n_steps, endpoint=False))
        return clear_graph(x)


def draw_t(key, batch, maxval=0.99):
    """
    One time step per graph, repeated over its edges.
    """

    t = jax.random.uniform(key, (batch.n_edge.shape[0], 1), maxval=maxval)
    return jnp.repeat(t, batch.n_edge, axis=0, total_repeat_length=batch.edges.shape[0])


def noisy_target(key, x, t):
    """
    Replaces the target slot (last on axis 1) of `[n, n_context + 1, d]` latents with its interpolation towards
    Gaussian noise and returns the target velocity.
    """

    z = jax.random.normal(key, (x.shape[0], x.shape[-1]), dtype=x.dtype)
    target = x[:, -1]
    return x.at[:, -1].set((1 - t) * z + t * target), target - z


def flow_matching_loss(model, params, key, batch, training=True, cond=None):
    """
    Masked MSE between the predicted and target velocities of the edge latents. Nodes and globals of the target
    slot are noise.
    """

    t_key, edges_key, nodes_key, globals_key, dropout_key = jax.random.split(key, 5)
    t = draw_t(t_key, batch)
    edges, v = noisy_target(edges_key, batch.edges, t)
    x = batch._replace(
        edges=edges,
        nodes=append_noise(nodes_key, batch.nodes[:, :-1]),
        globals=append_noise(globals_key, batch.globals[:, :-1])
    )
    v_pred = model.apply(params, x, t, training=training, cond=cond, rngs={'dropout': dropout_key}).edges
    mask = jraph.get_edge_padding_mask(batch)
    return (mask * ((v_pred - v) ** 2).mean(axis=-1)).sum() / mask.sum()


def append_noise(key, x):
    return jnp.concatenate([x, jax.random.normal(key, x[:, -1:].shape, dtype=x.dtype)], axis=1)


def context_slots(graph):
    """
    Drops the target slot of stacked `[n, n_context + 1, d]` features.
    """

    edges, nodes, globals_ = jax.tree.map(lambda x: x[:, :-1], (graph.edges, graph.nodes, graph.globals))
    return graph._replace(edges=edges, nodes=nodes, globals=globals_)


def stack_context(*graphs):
    """
    Stacks the latent graphs of contexts (and a target) on a new axis 1, as expected by `FlowMatching`.
    """

    return graphs[0]._replace(**{
        name: jnp.stack([getattr(g, name) for g in graphs], axis=1) for name in ('edges', 'nodes', 'globals')
    })


def latent_graph(context, z):
    return clear_graph(context._replace(edges=z))


def add_condition_input(params, edge_dim, n=1):
    """
    Adds `n` zero-initialised condition inputs to a pretrained model, so it starts from the unconditional one.
    """

    params = jax.tree.map(lambda x: x, params)
    dense = params['params']['Transformer_0']['Dense_0']
    dense['kernel'] = jnp.insert(dense['kernel'], edge_dim, jnp.zeros((n, dense['kernel'].shape[1])), axis=0)
    return params


def compute_mmd(x, y, x_mask, y_mask, gammas=None):
    """
    Maximum mean discrepancy with a multiscale RBF kernel (median heuristic) between masked sets of vectors.
    """

    def sq_distances(a, b):
        return (a ** 2).sum(-1, keepdims=True) + (b ** 2).sum(-1, keepdims=True).T - 2 * a @ b.T

    def kernel(a, b):
        return jax.vmap(lambda g: jnp.exp(-g * sq_distances(a, b)))(gammas).mean(axis=0)

    n, m = x_mask.sum(), y_mask.sum()

    if gammas is None:
        combined = jnp.concatenate([x, y], axis=0)
        dists = jnp.sqrt(jnp.maximum(sq_distances(combined, combined), 0.))
        dists = jnp.where(jnp.triu(jnp.ones_like(dists), k=1) > 0, dists, jnp.nan)
        gammas = jnp.asarray([0.125, 0.25, 0.5, 1., 2., 4., 8.]) / (2. * jnp.maximum(jnp.nanmedian(dists) ** 2, 1e-10))

    k_xx, k_yy, k_xy = kernel(x, x), kernel(y, y), kernel(x, y)
    s_xx = (k_xx * x_mask[:, None] * x_mask).sum() - (jnp.diag(k_xx) * x_mask).sum()
    s_yy = (k_yy * y_mask[:, None] * y_mask).sum() - (jnp.diag(k_yy) * y_mask).sum()
    s_xy = (k_xy * x_mask[:, None] * y_mask).sum()
    mmd = s_xx / (n * (n - 1)) + s_yy / (m * (m - 1)) - 2 * s_xy / (n * m)
    return jnp.sqrt(jnp.maximum(mmd, 0.))
