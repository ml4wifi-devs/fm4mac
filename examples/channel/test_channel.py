import os
os.environ['JAX_ENABLE_X64'] = 'True'

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.channel.baselines import blocks_overlap, coloring_plan, conflict_graph
from examples.channel.env import N_ACTIONS, decode_action, encode_action, evaluate, get_deployment
from examples.channel.graph import ChannelEdge, EdgeType, build_topology, neighbourhoods, plan_graph, to_graph
from examples.channel.pipeline import (
    ROOT,
    create_refiner,
    create_sampler,
    create_scorer,
    load_flow_matching,
    load_surrogate
)
from examples.wlan_sim.sim import SimParams
from fm4mac.autoencoder import Autoencoder
from fm4mac.flow import FlowMatching, stack_context
from fm4mac.graphs import stack_graphs, take_graphs
from fm4mac.surrogate import Surrogate


class ChannelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = get_deployment('random/6')
        cls.topology = build_topology(*cls.deployment)
        cls.plans = np.random.default_rng(0).integers(0, N_ACTIONS, (5, cls.topology.n_aps))

    def test_actions(self):
        primary, width = decode_action(np.arange(N_ACTIONS))
        self.assertEqual(primary.max(), 7)
        self.assertEqual(width.max(), 3)
        np.testing.assert_array_equal(encode_action(primary, width), np.arange(N_ACTIONS))

    def test_topology(self):
        t = self.topology
        self.assertTrue((t.type[:t.n_aps] == EdgeType.SELF.value).all())
        self.assertTrue((t.type[t.n_aps:] != EdgeType.SELF.value).all())
        self.assertEqual(t.n_nodes, t.n_aps * 3)
        neighbours = neighbourhoods(t)
        np.testing.assert_array_equal(neighbours, neighbours.T)
        np.testing.assert_array_equal(np.diag(neighbours), 1.)

    def test_graphs(self):
        graph = to_graph(self.topology, *decode_action(self.plans[0]), np.ones(self.topology.n_aps), xp=np)
        np.testing.assert_array_equal(graph.edges.primary[:self.topology.n_aps].argmax(-1), self.plans[0] // 4)
        self.assertTrue((graph.edges.width[self.topology.n_aps:, -1] == 1).all())

        batched = plan_graph(self.topology, *decode_action(self.plans))
        single = plan_graph(self.topology, *decode_action(jnp.asarray(self.plans[2])), xp=jnp)
        self.assertEqual(batched.nodes.shape, (5, self.topology.n_nodes, 0))
        np.testing.assert_allclose(batched.edges[2], single.edges)

    def test_coloring(self):
        ap_pos, sta_pos, _ = self.deployment
        conflict, _ = conflict_graph(ap_pos, sta_pos)
        primary, width = decode_action(coloring_plan(ap_pos, sta_pos, strict=True))

        for i, j in zip(*np.nonzero(conflict)):
            if primary[i] != primary[j]:
                self.assertFalse(blocks_overlap(primary[i], width[i], primary[j], width[j]))

    def test_evaluate(self):
        ap_pos, sta_pos, sta_mask = self.deployment
        params = SimParams(sim_time=20_000, dtype='float32')
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        thr = evaluate(keys, ap_pos[:2], sta_pos[:2], sta_mask[:2], self.plans[:2, :2], params)
        self.assertEqual(thr.shape, (2, 2))
        self.assertTrue((thr >= 0).all() and (thr > 0).any())

    def test_pipeline(self):
        key = jax.random.PRNGKey(0)
        n_aps, n_samples = self.topology.n_aps, 4
        autoencoder = Autoencoder(ChannelEdge, 8, (16,), (16,), 4, 1, jnp.float64)
        graph = to_graph(self.topology, *decode_action(self.plans[0]), np.ones(n_aps))
        ae_params = autoencoder.init({'params': key, 'sample': key}, graph)
        decode_fn = jax.jit(lambda z: autoencoder.apply(ae_params, z, method='decode'))

        encoded = autoencoder.apply(ae_params, graph, method='encode')
        context = stack_context(encoded)
        contexts = take_graphs(stack_graphs([context]), np.zeros(n_samples, dtype=int))
        fm = FlowMatching(1, 8, 16, 1, 2, 0., 4, jnp.float32)
        target = stack_context(encoded, encoded)
        fm_params = fm.init({'params': key, 'dropout': key}, target, jnp.zeros((target.edges.shape[0], 1)))
        z, plans = create_sampler(fm, decode_fn, n_aps, 2, n_steps=2)(fm_params, key, key, contexts)
        self.assertEqual(z.shape, (n_samples,) + encoded.edges.shape)
        self.assertEqual(plans.shape, (n_samples, n_aps))
        self.assertTrue(((plans >= 0) & (plans < N_ACTIONS)).all())

        surrogate = Surrogate(1, 8, 16, 1, 2, 0., jnp.float32, embed_nodes=False)
        s_params = surrogate.init(key, plan_graph(self.topology, *decode_action(self.plans[0])), training=False)
        score = create_scorer(surrogate, self.topology, 16)
        refined, values = create_refiner(score, n_aps)(s_params, self.plans, np.random.default_rng(0), 1)
        np.testing.assert_allclose(values, score(s_params, jnp.asarray(refined)), rtol=1e-5)
        self.assertTrue((values >= np.asarray(score(s_params, jnp.asarray(self.plans))) - 1e-5).all())

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, 'runs', 'rl_8')), 'shipped checkpoints not found')
    def test_checkpoints(self):
        fm, params, autoencoder_path = load_flow_matching('runs/rl_8')
        self.assertEqual(autoencoder_path, 'runs/autoencoder')
        self.assertEqual(fm.dim, 72)
        surrogate, s_params = load_surrogate('runs/rl_8')
        scores = create_scorer(surrogate, self.topology, 8)(s_params, jnp.asarray(self.plans))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == '__main__':
    unittest.main()
