import os
os.environ['JAX_ENABLE_X64'] = 'True'

import unittest

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from fm4mac.autoencoder import Autoencoder, autoencoder_loss, mixture_categorical_loss, squared_error
from fm4mac.flow import FlowMatching, add_condition_input, compute_mmd, flow_matching_loss, stack_context
from fm4mac.graphs import JaxEnum, edge_attributes, head_sizes, make_batch, pad_graph, stack_graphs, unbatch
from fm4mac.metrics import gm_ratio, pf_utility
from fm4mac.rl import create_nft_loss, group_advantages, local_utilities, nft_rewards
from fm4mac.search import create_coordinate_ascent, rank_correlation, select_covering
from fm4mac.surrogate import Surrogate, surrogate_loss, utility


class Kind(JaxEnum):
    A = 0
    B = 1
    NA = 2


@edge_attributes
class Edge:
    kind: Kind
    value: float


def ring(n, seed=0):
    rng = np.random.default_rng(seed)
    senders = np.arange(n)
    receivers = (senders + 1) % n
    edges = Edge(kind=np.eye(3)[rng.integers(0, 3, n)], value=rng.normal(size=(n, 1)))
    return jraph.GraphsTuple(
        nodes=np.zeros((n, 0)), edges=edges, senders=senders, receivers=receivers, globals=None,
        n_node=np.asarray([n]), n_edge=np.asarray([n])
    )


def flat(graph):
    return graph._replace(edges=np.concatenate([graph.edges.kind, graph.edges.value], axis=-1))


class GraphsTest(unittest.TestCase):
    def test_head_sizes(self):
        self.assertEqual(head_sizes(Edge), {'kind': 3, 'value': 1})

    def test_batch_roundtrip(self):
        graphs = [flat(ring(3)), flat(ring(5, 1))]
        batch = make_batch(graphs)
        self.assertEqual(batch.n_node.shape[0], 3)
        self.assertEqual(batch.senders.shape[0], 8)
        out = unbatch(batch)
        np.testing.assert_allclose(out[1].edges, graphs[1].edges)

    def test_pad_graph(self):
        graph = pad_graph(flat(ring(3)), 8, 8, np)
        self.assertEqual(graph.edges.shape, (8, 4))
        np.testing.assert_array_equal(graph.n_edge, [3, 5])


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.batch = make_batch([ring(4), ring(6, 1)])

    def test_autoencoder(self):
        model = Autoencoder(Edge, 8, (16,), (16,), 4, 2, jnp.float64)
        batch = self.batch._replace(edges=jax.tree.map(jnp.asarray, self.batch.edges))
        x = batch._replace(edges=jnp.concatenate([batch.edges.kind, batch.edges.value], axis=-1))
        params = model.init({'params': self.key, 'sample': self.key}, x)
        pred, z_mean, z_log_var = model.apply(params, x, rngs={'sample': self.key})
        self.assertEqual(pred.edges.kind.shape, (x.edges.shape[0], 3))
        self.assertEqual(z_mean.edges.shape, (x.edges.shape[0], 4))
        losses = {'kind': mixture_categorical_loss, 'value': squared_error}
        loss, kl, _ = autoencoder_loss(pred, batch, z_mean, z_log_var, losses, [1., 1.], 0.01)
        self.assertTrue(np.isfinite(loss) and kl >= 0)

    def latents(self, n_ctx=1, d=4):
        x = self.batch._replace(edges=jnp.ones((self.batch.edges.kind.shape[0], d)))
        x = x._replace(nodes=jnp.zeros((x.nodes.shape[0], d)), globals=jnp.zeros((x.n_node.shape[0], d)))
        return stack_context(*[x] * (n_ctx + 1))

    def test_flow_matching(self):
        model = FlowMatching(1, 8, 16, 1, 2, 0., 4, jnp.float64)
        x = self.latents()
        t = jnp.zeros((x.edges.shape[0], 1))
        params = model.init({'params': self.key, 'dropout': self.key}, x, t)
        loss = flow_matching_loss(model, params, self.key, x)
        self.assertTrue(np.isfinite(loss))

        context = x._replace(edges=x.edges[:, :1], nodes=x.nodes[:, :1], globals=x.globals[:, :1])
        out = model.apply(params, context, method='gen', rngs={'sample': self.key})
        self.assertEqual(out.edges.shape, (x.edges.shape[0], 4))

        cond = jnp.ones((x.edges.shape[0], 1))
        cond_params = add_condition_input(params, 8)
        v = model.apply(params, x, t, training=False).edges
        v_cond = model.apply(cond_params, x, t, training=False, cond=cond).edges
        np.testing.assert_allclose(v, v_cond, atol=1e-10)

        nft = create_nft_loss(model, 1., 5., 1e-4)
        contexts = stack_graphs([context, context])
        z = jnp.stack([out.edges, out.edges])
        loss, kl = nft(params, self.key, contexts, z, jnp.full((2, z.shape[1]), 0.7), params, params)
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(float(kl), 0.)

    def test_surrogate(self):
        graph = flat(ring(5))
        graphs = stack_graphs([jax.tree.map(jnp.asarray, graph)] * 4)
        graphs = graphs._replace(
            senders=graph.senders, receivers=graph.receivers, n_node=graph.n_node, n_edge=graph.n_edge,
            nodes=jnp.zeros((4, 5, 1))
        )
        for embed_nodes in (True, False):
            model = Surrogate(2, 8, 16, 1, 2, 0., jnp.float64, embed_nodes)
            params = model.init(self.key, graph._replace(nodes=np.zeros((5, 1))))
            targets, mask = jnp.ones((4, 3)), jnp.ones((4, 3))
            loss, (nll, rank, pred, true) = surrogate_loss(model, params, graphs, targets, mask, jnp.zeros(4), 1.)
            self.assertTrue(np.isfinite(loss))
            np.testing.assert_allclose(true, 3.)


class SearchTest(unittest.TestCase):
    def test_coordinate_ascent(self):
        weights = jnp.asarray([[1., -1., 2.], [0., 3., -2.]])
        options = np.tile(np.arange(3), (2, 1))
        search = create_coordinate_ascent(lambda w, c: w[jnp.arange(2), c].sum(-1))
        decisions, scores = search(weights, np.zeros((3, 2), dtype=int), options, True, np.random.default_rng(0))
        np.testing.assert_array_equal(decisions, [[2, 1]] * 3)
        np.testing.assert_allclose(scores, 5.)

        valid = np.ones((3, 2, 3), dtype=bool)
        valid[0, 0, 2] = False
        decisions, _ = search(weights, np.zeros((3, 2), dtype=int), options, valid, np.random.default_rng(0))
        np.testing.assert_array_equal(decisions[0], [0, 1])

    def test_select_covering(self):
        scores = np.asarray([4., 3., 2., 1.])
        covers = np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        np.testing.assert_array_equal(select_covering(scores, covers, 3), [0, 2, 3])
        np.testing.assert_array_equal(select_covering(scores, covers, 3, 0.), [0, 1, 2])

    def test_rank_correlation(self):
        a = np.asarray([[1., 2., 3.]])
        self.assertAlmostEqual(rank_correlation(a, a), 1.)
        self.assertAlmostEqual(rank_correlation(a, -a), -1.)


class RlTest(unittest.TestCase):
    def test_advantages(self):
        values = np.asarray([1., 3., 10., 10.])
        adv = group_advantages(values, 2)
        np.testing.assert_allclose(adv[2:], 0.)
        self.assertLess(adv[0], 0.)
        rewards = nft_rewards(adv, 5.)
        self.assertTrue(((rewards >= 0) & (rewards <= 1)).all())

        per_device = np.asarray([[1., 2.], [3., 4.], [5., 6.], [5., 6.]])
        adv = group_advantages(local_utilities(per_device, np.eye(2)), 2)
        self.assertEqual(adv.shape, (4, 2))

    def test_metrics(self):
        thr = np.asarray([[1., 4.], [2., 2.]])
        np.testing.assert_allclose(pf_utility(thr), [np.log(4.), 2 * np.log(2.)])
        np.testing.assert_allclose(gm_ratio(pf_utility(thr[0]), pf_utility(thr[1]), 2), 1.)
        np.testing.assert_allclose(utility(jnp.ones((2, 4)), jnp.ones((2, 3))), 3.)

    def test_mmd(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (64, 2))
        y = x + 5.
        mask = jnp.ones(64)
        self.assertLess(compute_mmd(x, x[::-1], mask, mask), compute_mmd(x, y, mask, mask))


if __name__ == '__main__':
    unittest.main()
