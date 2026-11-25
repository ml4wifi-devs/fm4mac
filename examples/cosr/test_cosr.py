import os
os.environ['JAX_ENABLE_X64'] = 'True'

import unittest

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from examples.cosr.env import (
    TEST_SCENARIOS,
    create_simulator,
    designated_links,
    random_choice,
    random_tx,
    refinement_options,
    round_robin_designated
)
from examples.cosr.graph import (
    choice_to_arrays,
    create_structure,
    decide,
    decode_decisions,
    designated_edges,
    encode_decisions,
    explicit_graph,
    network_graph,
    selected_to_tx
)
from examples.cosr.pipeline import Pipeline, create_refiner, path, surrogate_scores
from fm4mac.graphs import nearest_power_of_two, pad_graph
from fm4mac.surrogate import Surrogate


class CoSRTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = TEST_SCENARIOS['residential'][1]
        cls.st = create_structure(cls.scenario)
        cls.rng = np.random.default_rng(0)
        cls.decisions = random_choice(cls.rng, cls.st, 16)
        cls.arrays = [np.asarray(a) for a in choice_to_arrays(cls.st, *map(jnp.asarray, cls.decisions))]
        cls.model = Surrogate(2, 32, 64, 1, 4, 0., jnp.float32)
        graph = explicit_graph(cls.st, *[a[0] for a in cls.arrays])
        cls.params = cls.model.init(jax.random.PRNGKey(0), graph, training=False)

    def test_structure(self):
        n_links = len(self.st.link_ap)
        self.assertEqual(n_links, sum(len(s) for s in self.scenario.associations.values()))
        np.testing.assert_array_equal(self.st.senders[:n_links], self.st.link_ap)
        np.testing.assert_array_equal(self.st.receivers[:n_links], self.st.link_sta)
        np.testing.assert_array_equal(self.st.link_ap[self.st.ap_links], np.repeat(self.st.ap_ids[:, None], 4, axis=1))

    def test_graphs_agree(self):
        for i in range(4):
            graph = network_graph(self.st, *[a[i] for a in self.arrays])
            edges = np.asarray(explicit_graph(self.st, *[a[i] for a in self.arrays]).edges)
            np.testing.assert_allclose(edges[:, :2], graph.edges.type[:, :2])
            np.testing.assert_allclose(edges[:, 3], graph.edges.rssi[:, 0], rtol=1e-6)
            np.testing.assert_allclose(edges[:, 4], graph.edges.selected[:, 1])
            np.testing.assert_allclose(edges[:, 5:9], graph.edges.tx_power[:, :4])
            np.testing.assert_allclose(edges[:, 9:23], graph.edges.mcs[:, :14])

        batch = explicit_graph(self.st, *self.arrays)
        single = jax.vmap(lambda *a: explicit_graph(self.st, *a).edges)(*self.arrays)
        np.testing.assert_allclose(np.asarray(batch.edges), np.asarray(single))

    def test_conversions(self):
        choice, power, mcs = self.decisions
        tx = np.asarray(selected_to_tx(self.st, self.arrays[0]))
        np.testing.assert_array_equal(tx[:, self.st.link_ap, self.st.link_sta], self.arrays[0])
        np.testing.assert_array_equal(tx.sum(axis=(-2, -1)), (choice >= 0).sum(axis=-1))

        decoded = [np.asarray(a) for a in decode_decisions(encode_decisions(*map(jnp.asarray, self.decisions)))]
        active = choice >= 0
        np.testing.assert_array_equal(decoded[0], choice)
        np.testing.assert_array_equal(decoded[1][active], power[active])
        np.testing.assert_array_equal(decoded[2][active], mcs[active])

    def test_decide(self):
        choice, power, mcs = (a[0] for a in self.decisions)
        graph = network_graph(self.st, *[a[0] for a in self.arrays])
        edges = jax.tree.map(lambda x: 50. * jnp.asarray(x), graph.edges)
        designated = round_robin_designated(jax.random.PRNGKey(0), self.scenario.associations, 2, 1)[0]
        none = jnp.full((1, 2), -1)

        out = [np.asarray(a) for a in decide(jax.random.PRNGKey(1), edges, self.st, none)]
        np.testing.assert_array_equal(out[0], choice)
        np.testing.assert_array_equal(out[1][choice >= 0], power[choice >= 0])
        np.testing.assert_array_equal(out[2][choice >= 0], mcs[choice >= 0])

        out = decide(jax.random.PRNGKey(1), edges, self.st, designated)[0]
        fixed = designated_links(self.st, designated[None])[0]
        self.assertEqual(int((fixed >= 0).sum()), 2)
        np.testing.assert_array_equal(np.asarray(out)[fixed >= 0], fixed[fixed >= 0])

        cond = np.asarray(designated_edges(self.st, designated))[:, 0]
        self.assertEqual(cond.sum(), 4)
        np.testing.assert_array_equal(np.flatnonzero(cond[:len(self.st.link_ap)]), np.sort(fixed[fixed >= 0]))

    def test_random_tx(self):
        choice, power, mcs = (np.asarray(a) for a in random_tx(jax.random.PRNGKey(0), self.st))
        own = (self.st.ap_links == choice[:, None]).any(axis=-1)
        self.assertTrue((own | (choice < 0)).all())

    def test_refinement_options(self):
        fixed = np.full((2, len(self.st.ap_ids)), -1)
        fixed[1, 0] = self.st.ap_links[0, 1]
        options, valid = refinement_options(self.st, fixed)
        links = decode_decisions(jnp.asarray(options))[0]
        self.assertEqual(options.shape[1], 1 + 4 * 4 * 14)
        self.assertTrue(valid[0].all())
        np.testing.assert_array_equal(np.asarray(links)[0][valid[1, 0]], self.st.ap_links[0, 1])

    def test_link_rates_sum_to_aggregate(self):
        thr, link_rates = create_simulator(self.scenario, self.st)(jax.random.PRNGKey(1), *self.decisions)
        np.testing.assert_allclose(link_rates.sum(axis=-1), thr)
        self.assertTrue((link_rates[self.arrays[0] == 0] == 0).all())

    def test_padding_keeps_predictions(self):
        graph = explicit_graph(self.st, *[a[0] for a in self.arrays])
        mean, _ = self.model.apply(self.params, graph, training=False)
        sizes = nearest_power_of_two(len(self.st.is_ap) + 1), nearest_power_of_two(len(self.st.senders))
        padded = pad_graph(graph, *sizes)
        padded_mean, _ = self.model.apply(self.params, padded, training=False)
        np.testing.assert_allclose(np.asarray(padded_mean)[:len(mean)], np.asarray(mean), atol=1e-2)
        self.assertEqual(int(jraph.get_edge_padding_mask(padded).sum()), len(mean))

    def test_refiner(self):
        choice, power, mcs = self.decisions
        fixed = np.full_like(choice, -1)
        fixed[:, 0] = self.st.ap_links[0, 1]
        choice = np.where(fixed >= 0, fixed, choice)
        score = lambda *d: np.asarray(
            surrogate_scores(self.model, self.params, self.st, *choice_to_arrays(self.st, *map(jnp.asarray, d)), 64)
        )

        before = score(choice, power, mcs)
        *refined, scores = create_refiner(self.model, 256)(
            self.params, self.st, choice, power, mcs, fixed, np.random.default_rng(0)
        )
        after = score(*refined)
        np.testing.assert_array_equal(refined[0][:, 0], fixed[:, 0])
        np.testing.assert_allclose(after, scores, rtol=1e-3, atol=1e-2)
        self.assertTrue((after >= before - 1e-2).all())
        self.assertTrue((after > before + 1e-3).any())

    @unittest.skipUnless(os.path.exists(path('runs/surrogate')), 'shipped runs not found')
    def test_shipped_pipeline(self):
        scenario = TEST_SCENARIOS['residential'][0]
        st = create_structure(scenario)
        pipe = Pipeline('runs/flow_matching', 'runs/surrogate', chunk_size=4)
        contexts = pipe.contexts(jax.random.PRNGKey(0), scenario, st, 4)
        designated = round_robin_designated(jax.random.PRNGKey(1), scenario.associations, 1, 4)
        keys = jax.random.split(jax.random.PRNGKey(2))
        _, *decisions = pipe.generate(*keys, pipe.params, st, contexts, designated)
        fixed = designated_links(st, designated)
        np.testing.assert_array_equal(decisions[0][fixed >= 0], fixed[fixed >= 0])
        self.assertEqual(pipe.scores(pipe.s_params, st, *decisions).shape, (4,))


if __name__ == '__main__':
    unittest.main()
