import os
os.environ['JAX_ENABLE_X64'] = 'True'

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.wlan_sim.constants import (
    ACK_DURATION,
    AM,
    CTS_DURATION,
    DIFS,
    MAX_PPDU_DURATION,
    N_CHANNELS,
    N_WIDTHS,
    OFDM_SYMBOL,
    OP,
    RTS_DURATION,
    SCB,
    SIFS,
    SLOT
)
from examples.wlan_sim.komondor import KOMONDOR_DIR, run
from examples.wlan_sim.phy import (
    bits_per_symbol,
    data_duration,
    fit_aggregated,
    link_tables,
    max_aggregated,
    path_gain,
    path_loss
)
from examples.wlan_sim.sim import SimParams, select_channels, simulate
from examples.wlan_sim.validate import jax_throughput


def single_wlan(width, distance=3.):
    ap_pos = jnp.array([[0., 0., 0.]])
    sta_pos = jnp.array([[[0., distance, 0.]]])
    return ap_pos, sta_pos, jnp.ones((1, 1), dtype=bool), jnp.array([0]), jnp.array([width]), jnp.array([AM])


def simulate_keys(keys, *args):
    return jax.vmap(simulate, in_axes=(0,) + (None,) * len(args))(keys, *args)


class PhyTest(unittest.TestCase):
    def test_path_loss(self):
        # Received powers logged by Komondor at 20 dBm
        self.assertAlmostEqual(20. - float(path_loss(3.)), -55.77, places=2)
        self.assertAlmostEqual(20. - float(path_loss(2.)), -44.97, places=2)
        self.assertAlmostEqual(20. - 3.01 - float(path_loss(4.)), -67.66, places=1)

    def test_rates(self):
        expected = {0: (7.31, 14.62, 30.62, 61.25), 5: (43.88, 87.75, 183.75, 367.5), 8: (87.75, 175.5, 367.5, 735.),
                    11: (121.88, 243.75, 510.42, 1020.83), 13: (146.25, 292.5, 612.5, 1225.)}
        for mcs, rates in expected.items():
            rate = bits_per_symbol(jnp.array(mcs), jnp.arange(N_WIDTHS)) / OFDM_SYMBOL
            np.testing.assert_allclose(rate, rates, atol=0.01)

    def test_aggregation_limit(self):
        # MCS 0 at 160 MHz fits 26 MPDUs of 12000 bits in 5.484 ms (Komondor log)
        self.assertEqual(int(max_aggregated(bits_per_symbol(0, 3), 64, 12000)), 26)
        self.assertEqual(int(max_aggregated(bits_per_symbol(13, 3), 64, 12000)), 64)

    def test_fit_aggregated(self):
        # Komondor drops one aggregated packet at a time until the DATA fits
        for mcs in range(14):
            for width in range(N_WIDTHS):
                bps = bits_per_symbol(mcs, width)
                for limit in (500, 1000, 2500, MAX_PPDU_DURATION):
                    n = 64
                    while n > 1 and int(data_duration(n, bps, 12000)) > limit:
                        n -= 1
                    self.assertEqual(min(int(fit_aggregated(bps, limit, 12000)), 64), n)

    def test_link_tables(self):
        gain = path_gain(jnp.array([[0., 0., 0.], [0., 3., 0.]]))
        _, mcs, n, duration, announced = link_tables(gain, jnp.array([20.]), 1, 1, 64, 12000)
        # RSSI -55.77 dBm: >= -57 (MCS 9) at 20 MHz, >= -59 + 3 (MCS 8) at 40 MHz, >= -64 + 6 (MCS 7) at 80 MHz
        np.testing.assert_array_equal(mcs[0, 0], [9, 8, 7, 6])
        # DATA narrowed by the CTS fits in the duration announced by the RTS
        rts_duration = np.diagonal(announced[0, 0])
        self.assertTrue((duration[0, 0] <= rts_duration[:, None]).all())
        self.assertTrue((n[0, 0] <= np.diagonal(n[0, 0])[:, None]).all())
        np.testing.assert_array_equal(np.diagonal(duration[0, 0]), rts_duration)
        self.assertTrue((announced[0, 0] >= duration[0, 0]).all())


class ChannelSelectionTest(unittest.TestCase):
    params = SimParams()
    key = jax.random.PRNGKey(0)

    def select(self, power, primary, width, policy):
        possible, level = select_channels(jnp.array(power, dtype=float), primary, width, policy, self.key, self.params)
        return bool(possible), int(level)

    def test_idle(self):
        idle = [-100.] * N_CHANNELS
        self.assertEqual(self.select(idle, 5, 3, OP), (True, 0))
        self.assertEqual(self.select(idle, 5, 3, SCB), (True, 3))
        self.assertEqual(self.select(idle, 5, 3, AM), (True, 3))
        self.assertEqual(self.select(idle, 5, 2, AM), (True, 2))

    def test_always_max(self):
        power = [-100.] * N_CHANNELS
        power[0] = -70.
        # Secondary 80 MHz busy above -66 dBm only
        self.assertEqual(self.select(power, 5, 3, AM), (True, 3))
        power[0] = -60.
        self.assertEqual(self.select(power, 5, 3, AM), (True, 2))
        # Secondary 40 MHz busy above -69 dBm
        power[6] = -68.
        self.assertEqual(self.select(power, 5, 3, AM), (True, 1))
        # Secondary 20 MHz busy above -72 dBm, AM still transmits on the primary
        power[4] = -71.
        self.assertEqual(self.select(power, 5, 3, AM), (True, 0))

    def test_static(self):
        power = [-100.] * N_CHANNELS
        power[7] = -71.
        self.assertEqual(self.select(power, 5, 3, SCB)[0], False)
        self.assertEqual(self.select(power, 5, 1, SCB), (True, 1))
        power[5] = -81.
        self.assertEqual(self.select(power, 5, 1, OP)[0], False)


class SimulatorTest(unittest.TestCase):
    def analytic(self, width, rts_cts, params):
        _, _, n, duration, _ = link_tables(path_gain(jnp.array([[0., 0., 0.], [0., 3., 0.]])), jnp.array([20.]), 1, 1,
                                           params.n_agg, params.packet_length)
        backoff = (params.cw_min + params.cw_max) / 2 + 1
        exchange = int(duration[0, 0, width, width]) + SIFS + ACK_DURATION
        if rts_cts:
            exchange += RTS_DURATION + CTS_DURATION + 2 * SIFS
        return int(n[0, 0, width, width]) * params.packet_length / (DIFS + backoff * SLOT + exchange)

    def test_single_wlan(self):
        for width in range(N_WIDTHS):
            for rts_cts in (False, True):
                params = SimParams(sim_time=5_000_000, rts_cts=rts_cts)
                thr, _ = simulate_keys(jax.random.split(jax.random.PRNGKey(width), 8), *single_wlan(width), params)
                self.assertAlmostEqual(float(thr.mean()) / self.analytic(width, rts_cts, params), 1., delta=0.005)

    def test_disjoint_channels(self):
        params = SimParams(sim_time=2_000_000)
        ap_pos = jnp.array([[0., 0., 0.], [3., 0., 0.]])
        sta_pos = jnp.array([[[0., 3., 0.]], [[3., 3., 0.]]])
        mask = jnp.ones((2, 1), dtype=bool)
        key = jax.random.PRNGKey(0)
        thr, _ = simulate(key, ap_pos, sta_pos, mask, jnp.array([0, 4]), jnp.array([2, 2]), jnp.array([AM, AM]), params)
        alone, _ = simulate(key, *single_wlan(2), params)
        np.testing.assert_allclose(thr, float(alone[0]), rtol=0.02)

    def test_equal_share(self):
        params = SimParams(sim_time=5_000_000)
        ap_pos = jnp.array([[0., 0., 0.], [1., 0., 0.]])
        sta_pos = jnp.array([[[0., 1., 0.]], [[1., 1., 0.]]])
        thr, _ = simulate(jax.random.PRNGKey(0), ap_pos, sta_pos, jnp.ones((2, 1), dtype=bool), jnp.array([0, 0]),
                          jnp.array([0, 0]), jnp.array([AM, AM]), params)
        self.assertAlmostEqual(float(thr[0] / thr[1]), 1., delta=0.05)

    def test_per_sta_matches_per_ap(self):
        params = SimParams(sim_time=2_000_000)
        rng = np.random.default_rng(0)
        ap_pos = np.concatenate([rng.uniform(0, 20, (4, 2)), np.zeros((4, 1))], axis=1)
        sta_pos = ap_pos[:, None] + np.concatenate([rng.uniform(-4, 4, (4, 3, 2)), np.zeros((4, 3, 1))], axis=-1)
        args = (ap_pos, sta_pos, np.ones((4, 3), dtype=bool), np.array([0, 1, 4, 2]), np.array([2, 1, 3, 0]),
                np.array([AM, SCB, AM, OP]))
        thr, state = simulate(jax.random.PRNGKey(0), *args, params)
        np.testing.assert_array_equal(state.sta_packets.sum(-1), state.n_packets)
        # An explicit identity slot layout is the default layout
        _, explicit = simulate(jax.random.PRNGKey(0), *args, params, 4 + jnp.arange(12).reshape(4, 3))
        np.testing.assert_array_equal(explicit.sta_packets, state.sta_packets)

    def test_silent_ap(self):
        # An AP without STAs does not contend: a co-channel neighbour gets the same throughput as when alone
        params = SimParams(sim_time=2_000_000)
        scenario = dict(
            ap_pos=np.array([[0., 0., 0.], [3., 0., 0.]]), sta_pos=np.array([[1., 1., 0.], [2., 0., 0.]]),
            serving=np.array([0, 0]), primary=np.array([0, 0]), width=np.array([0, 0]), policy=np.array([AM, AM])
        )
        keys = jax.random.split(jax.random.PRNGKey(1), 16)
        both_ap, both_sta = jax_throughput(keys, scenario, params)
        scenario |= dict(ap_pos=scenario['ap_pos'][:1], primary=scenario['primary'][:1], width=scenario['width'][:1],
                         policy=scenario['policy'][:1])
        alone_ap, alone_sta = jax_throughput(keys, scenario, params)
        np.testing.assert_array_equal(both_ap[:, 1], 0.)
        np.testing.assert_allclose(both_sta.mean(0), alone_sta.mean(0), rtol=0.05)
        np.testing.assert_allclose(both_ap.sum(-1), both_sta.sum(-1), rtol=1e-9)


@unittest.skipUnless(os.path.exists(os.path.join(KOMONDOR_DIR, 'main', 'komondor_main')), 'Komondor not available')
class KomondorTest(unittest.TestCase):
    def compare(self, scenario, params, n_seeds=4, rtol=0.05, atol=0.):
        komondor = [run(*scenario.values(), params, seed + 1) for seed in range(n_seeds)]
        thr = jax_throughput(jax.random.split(jax.random.PRNGKey(0), n_seeds), scenario, params)
        for k, j in zip(zip(*komondor), thr):
            np.testing.assert_allclose(j.mean(0), np.mean(k, axis=0), rtol=rtol, atol=atol)

    def test_overlapping_bonding(self):
        scenario = dict(
            ap_pos=np.array([[0., 0., 0.], [4., 0., 0.]]), sta_pos=np.array([[0., 3., 0.], [4., 3., 0.]]),
            serving=np.array([0, 1]), primary=np.array([0, 2]), width=np.array([2, 2]), policy=np.array([AM, AM])
        )
        for rts_cts in (False, True):
            self.compare(scenario, SimParams(sim_time=10_000_000, cw_adaptation=True, rts_cts=rts_cts))

    def test_hidden_station(self):
        scenario = dict(
            ap_pos=np.array([[3.6555, 16.1915, 0.], [0.8163, 12.8885, 0.]]),
            sta_pos=np.array([
                [5.0827, 15.9326, 0.], [2.8035, 13.2475, 0.], [-2.6016, 9.4255, 0.], [1.1161, 11.3766, 0.]
            ]),
            serving=np.array([0, 0, 1, 1]), primary=np.array([7, 4]), width=np.array([3, 3]), policy=np.array([AM, AM])
        )
        self.compare(scenario, SimParams(sim_time=10_000_000, cw_adaptation=True), n_seeds=6)

    def test_hidden_sta_association(self):
        # Two co-channel APs out of each other's range; STA 2 between them is associated to AP 0, so AP 1 is hidden
        scenario = dict(
            ap_pos=np.array([[0., 0., 0.], [9., 0., 0.]]),
            sta_pos=np.array([[2., 0., 0.], [0., 2., 0.], [4.5, 0., 0.], [9., 2., 0.]]),
            serving=np.array([0, 0, 0, 1]), primary=np.array([0, 0]), width=np.array([0, 0]), policy=np.array([AM, AM])
        )
        self.compare(scenario, SimParams(sim_time=5_000_000, cw_adaptation=True), n_seeds=4, rtol=0.1, atol=1.)


if __name__ == '__main__':
    unittest.main()
