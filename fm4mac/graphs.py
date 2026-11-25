import enum
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jraph
import numpy as np


@jax.tree_util.register_pytree_node_class
class JaxEnum(enum.Enum):
    """
    Categorical edge attribute: an enum member becomes its one-hot vector inside jax. The last member is the
    `NA` class used when the attribute does not apply to an edge.
    """

    def __jax_array__(self):
        return jax.nn.one_hot(self.value, len(type(self)))

    def __init_subclass__(cls) -> None:
        jax.tree_util.register_pytree_node_class(cls)
        return super().__init_subclass__()

    def tree_flatten(self):
        return (self.__jax_array__(),), self

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return children[0]


def edge_attributes(cls):
    """
    Registers a dataclass of edge attributes as a pytree. Fields annotated with a `JaxEnum` are categorical,
    the others are scalars.
    """

    return jax.tree_util.register_dataclass(dataclass(cls))


def head_sizes(cls):
    """
    Output size of the decoder head of every attribute, in field order.
    """

    hints = typing.get_type_hints(cls)
    return {name: len(t) if isinstance(t, type) and issubclass(t, JaxEnum) else 1 for name, t in hints.items()}


def is_categorical(cls, name):
    t = typing.get_type_hints(cls)[name]
    return isinstance(t, type) and issubclass(t, JaxEnum)


def nearest_power_of_two(n):
    return 2 ** (int(n) - 1).bit_length()


def make_batch(graphs, pad=True):
    """
    Batches graphs and pads them with one padding graph to power-of-two node and edge counts, which bounds the
    number of compilations. Drop the padding graph from per-graph outputs with `[:-1]`.
    """

    batch = jraph.batch_np(graphs)

    if pad:
        n_node = nearest_power_of_two(batch.n_node.sum() + 1)
        n_edge = nearest_power_of_two(batch.senders.shape[0])
        batch = jraph.pad_with_graphs(batch, n_node, n_edge, batch.n_node.shape[0] + 1)

    return batch


def unbatch(batch):
    return jraph.unbatch_np(jraph.unpad_with_graphs(batch))


def pad_graph(graph, n_node_pad, n_edge_pad, xp=jnp):
    """
    Pads a single graph (features may have leading batch axes) with one padding graph to fixed sizes.
    """

    n_node, n_edge = graph.nodes.shape[-2], graph.edges.shape[-2]
    pad_node, pad_edge = n_node_pad - n_node, n_edge_pad - n_edge
    batch = [(0, 0)] * (graph.edges.ndim - 2)
    return graph._replace(
        nodes=xp.pad(graph.nodes, batch + [(0, pad_node), (0, 0)]),
        edges=xp.pad(graph.edges, batch + [(0, pad_edge), (0, 0)]),
        senders=xp.pad(graph.senders, (0, pad_edge), constant_values=n_node),
        receivers=xp.pad(graph.receivers, (0, pad_edge), constant_values=n_node),
        n_node=xp.asarray([n_node, pad_node]),
        n_edge=xp.asarray([n_edge, pad_edge])
    )


def clear_graph(graph):
    """
    Keeps only the edge features: the latent space lives on edges.
    """

    return graph._replace(nodes=jnp.empty((graph.nodes.shape[0], 0), dtype=graph.nodes.dtype), globals=None)


def stack_graphs(graphs):
    return jax.tree.map(lambda *x: jnp.stack(x), *graphs)


def take_graphs(graphs, idxs):
    return jax.tree.map(lambda x: x[idxs], graphs)


def segment_ids(n_elements, total_length):
    return jnp.searchsorted(jnp.cumsum(n_elements), jnp.arange(total_length), side='right')


def segment_sum(graph, values, mask=None):
    """
    Per-graph sum of per-edge values (optionally masked), e.g. a utility from per-device predictions.
    """

    if mask is not None:
        values = jnp.where(mask, values, 0.)

    ids = segment_ids(graph.n_edge, graph.senders.shape[0])
    return jax.ops.segment_sum(values, ids, graph.n_edge.shape[0])


def bidirectional(senders, receivers):
    """
    Edge lists in both directions: forward edges first, then their reverses in the same order.
    """

    senders, receivers = np.asarray(senders), np.asarray(receivers)
    return np.concatenate([senders, receivers]), np.concatenate([receivers, senders])


SHARED_STRUCTURE = jraph.GraphsTuple(
    nodes=0, edges=0, senders=None, receivers=None, globals=None, n_node=None, n_edge=None
)
