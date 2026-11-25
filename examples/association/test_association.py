import os
os.environ['JAX_ENABLE_X64'] = 'True'

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.association.baselines import (
    AirtimeModel,
    create_local_search,
    estimated_throughput,
    model_pf,
    strongest_signal
)
from examples.association.env import (
    Deployment,
    evaluate,
    get_deployment,
    make_network,
    option_table,
    random_assoc
)
from examples.association.graph import (
    N_FEATURES,
    build_topology,
    plan_graph,
    sample_assoc,
    sta_targets,
    target_mask,
    to_graph
)
from examples.association.pipeline import create_refiner, create_scorer, plan_graphs
from examples.association.rl import sta_neighbourhoods
from examples.wlan_sim.sim import SimParams
from fm4mac.surrogate import Surrogate, apply_batch, utility


PARAMS = SimParams(sim_time=100_000, cw_adaptation=True, tie_order='random', dtype='float32')


class EnvTest(unittest.TestCase):
    deployment = get_deployment('random/6')
    net = make_network(deployment)

    def test_slots(self):
        cand, slot_sta = np.asarray(self.net.candidates), np.asarray(self.net.slot_sta)

        for a in range(cand.shape[1]):
            np.testing.assert_array_equal(np.sort(slot_sta[a][slot_sta[a] >= 0]), np.flatnonzero(cand[:, a]))

        table = option_table(self.net)
        self.assertTrue(cand[np.arange(len(cand))[:, None], np.maximum(table, 0)][table >= 0].all())
        np.testing.assert_array_equal((table >= 0).sum(1), cand.sum(1))

    def test_silent_ap(self):
        # An AP without STAs does not contend: a co-channel neighbour gets the same throughput as when alone
        sta_pos = np.array([[1., 1., 0.], [2., 0., 0.]])
        both = Deployment(np.array([[0., 0., 0.], [3., 0., 0.]]), sta_pos, np.array([0, 0]), np.array([0, 0]))
        alone = Deployment(np.array([[0., 0., 0.]]), sta_pos, np.array([0]), np.array([0]))
        keys = jax.random.split(jax.random.PRNGKey(1), 16)
        thr_both = np.asarray(evaluate(keys, make_network(both), jnp.zeros((16, 2), dtype=int))).mean(0)
        thr_alone = np.asarray(evaluate(keys, make_network(alone), jnp.zeros((16, 2), dtype=int))).mean(0)
        np.testing.assert_allclose(thr_both, thr_alone, rtol=0.05)


class GraphTest(unittest.TestCase):
    deployment = get_deployment('random/8')
    net = make_network(deployment)
    topology = build_topology(deployment)

    def test_roundtrip(self):
        assoc = random_assoc(np.random.default_rng(1), self.net)
        graph = to_graph(self.topology, jnp.asarray(self.deployment.primary), jnp.asarray(assoc))
        logits = graph.edges.assoc[self.topology.n_aps:self.topology.n_aps + self.topology.n_candidates]
        sampled = sample_assoc(jax.random.PRNGKey(0), 100. * (logits[:, 1] - logits[:, 0]), self.topology)
        np.testing.assert_array_equal(sampled, assoc)

    def test_plan_graph(self):
        assoc = strongest_signal(self.net)
        graph = plan_graph(self.topology, np.asarray(self.deployment.primary), assoc)
        self.assertEqual(graph.edges.shape, (len(self.topology.senders), N_FEATURES))
        mask = target_mask(self.topology, assoc)
        sta = self.topology.cand_sta[np.flatnonzero(mask) - self.topology.n_aps]
        np.testing.assert_array_equal(np.sort(sta), np.arange(len(assoc)))
        jgraph = plan_graph(self.topology, jnp.asarray(self.deployment.primary), jnp.asarray(assoc), xp=jnp)
        np.testing.assert_allclose(np.asarray(jgraph.edges), graph.edges)
        thr = np.arange(len(assoc)) + 1.
        np.testing.assert_allclose(utility(sta_targets(self.topology, thr), mask), np.log(thr).sum())


class BaselinesTest(unittest.TestCase):
    deployment = get_deployment('random/4')
    net = make_network(deployment)

    def test_heuristics(self):
        model, ssf, rng = AirtimeModel(self.net, PARAMS), strongest_signal(self.net), np.random.default_rng(0)
        cand = np.asarray(self.net.candidates)

        for assoc in (ssf, estimated_throughput(model, ssf, rng), model_pf(model, ssf, rng)):
            self.assertTrue(cand[np.arange(len(assoc)), assoc].all())

        self.assertGreaterEqual(model.utility(model_pf(model, ssf, rng)), model.utility(ssf))

    def test_local_search(self):
        table, ssf = option_table(self.net), strongest_signal(self.net)
        search = create_local_search(2, PARAMS)
        args = (self.net, jax.random.PRNGKey(0))
        options, rng = np.maximum(table, 0), np.random.default_rng(0)
        found, value = search(args, ssf[None], options, table >= 0, rng, 1)
        _, init_value = search(args, ssf[None], options, table == ssf[:, None], rng, 1)
        self.assertGreaterEqual(value[0], init_value[0])
        self.assertTrue(np.asarray(self.net.candidates)[np.arange(len(ssf)), found[0]].all())


class PipelineTest(unittest.TestCase):
    deployment = get_deployment('random/6')
    net = make_network(deployment)
    topology = build_topology(deployment)

    def test_scorer_and_refiner(self):
        primary = jnp.asarray(self.deployment.primary)
        model = Surrogate(1, 16, 32, 1, 2, 0., jnp.float32, embed_nodes=False)
        graph = plan_graph(self.topology, np.asarray(self.deployment.primary), strongest_signal(self.net))
        s_params = model.init(jax.random.PRNGKey(0), graph, training=False)
        rng = np.random.default_rng(0)
        plans = np.stack([random_assoc(rng, self.net) for _ in range(3)])
        score = create_scorer(model, self.topology, primary)
        scores = np.asarray(score(s_params, jnp.asarray(plans)))

        graphs, mask = plan_graphs(self.topology, primary, jnp.asarray(plans))
        mean, _ = apply_batch(model, s_params, graphs)
        np.testing.assert_allclose(utility(mean, mask), scores, rtol=1e-5)

        table = option_table(self.net)
        refined, values = create_refiner(score)(s_params, plans, np.maximum(table, 0), table >= 0, rng, 1)
        np.testing.assert_allclose(values, np.asarray(score(s_params, jnp.asarray(refined))), rtol=1e-5)
        self.assertTrue((values >= scores - 1e-5).all())

    def test_neighbourhoods(self):
        neighbourhoods = sta_neighbourhoods(self.deployment, self.net)
        self.assertTrue((neighbourhoods >= np.asarray(self.net.candidates)).all())


if __name__ == '__main__':
    unittest.main()
