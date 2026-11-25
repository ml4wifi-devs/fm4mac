from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from mapc_research.envs.scenario_impl import random_scenario, residential_scenario
from mapc_sim.constants import TAU

from examples.cosr.env import open_enterprise_scenario


@dataclass
class MobilityState:
    pos: np.ndarray
    velocity: np.ndarray
    remaining_time: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    mobile_mask: np.ndarray
    speed_min: float
    speed_max: float
    segment_min_s: float
    segment_max_s: float


def room_bounds(x_apartments, y_apartments, n_sta_per_ap, size):
    """
    Per-node bounds of the residential scenario (stations move within their rooms, APs are pinned) and the mobile mask,
    in the node order of `residential_scenario`.
    """

    bounds_min, bounds_max, mobile_mask = [], [], []

    for x, y in product(range(x_apartments), range(y_apartments)):
        bounds_min += [[x * size, y * size]] * (n_sta_per_ap + 1)
        bounds_max += [[(x + 1) * size, (y + 1) * size]] * (n_sta_per_ap + 1)
        mobile_mask += [False] + [True] * n_sta_per_ap

    return np.asarray(bounds_min, dtype=float), np.asarray(bounds_max, dtype=float), np.asarray(mobile_mask)


def area_bounds(scenario, bounds):
    """
    Bounds of the random scenario: stations move freely within [0, bounds]^2, APs are pinned.
    """

    n_nodes = len(scenario.pos)
    mobile_mask = ~np.isin(np.arange(n_nodes), list(scenario.associations))
    return np.zeros((n_nodes, 2)), np.full((n_nodes, 2), bounds, dtype=float), mobile_mask


def sample_velocities(key, n_nodes, speed_min, speed_max):
    angle_key, speed_key = jax.random.split(key)
    angles = np.asarray(jax.random.uniform(angle_key, (n_nodes,), minval=0., maxval=2 * jnp.pi))
    speeds = np.asarray(jax.random.uniform(speed_key, (n_nodes,), minval=speed_min, maxval=speed_max))
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1) * speeds[:, None]


def sample_segment_times(key, n_nodes, segment_min_s, segment_max_s):
    return np.asarray(jax.random.uniform(key, (n_nodes,), minval=segment_min_s, maxval=segment_max_s), dtype=float)


def create_mobility_state(key, initial_pos, bounds_min, bounds_max, mobile_mask, speed_min, speed_max,
                          segment_min_s, segment_max_s):
    pos = np.clip(np.asarray(initial_pos, dtype=float), bounds_min, bounds_max)
    velocity_key, time_key = jax.random.split(key)
    velocity = sample_velocities(velocity_key, len(pos), speed_min, speed_max)
    velocity[~mobile_mask] = 0.
    return MobilityState(
        pos, velocity, sample_segment_times(time_key, len(pos), segment_min_s, segment_max_s), bounds_min, bounds_max,
        mobile_mask, speed_min, speed_max, segment_min_s, segment_max_s
    )


def step_mobility_state(key, state, dt):
    """
    Random waypoint-like mobility: nodes move with constant velocity for a random time and draw a new velocity when
    the segment ends or they hit their bounds.
    """

    proposed = state.pos + state.velocity * dt
    pos = np.clip(proposed, state.bounds_min, state.bounds_max)
    remaining_time = state.remaining_time - dt
    redraw = (np.any(np.abs(proposed - pos) > 1e-12, axis=1) | (remaining_time <= 0.)) & state.mobile_mask

    if np.any(redraw):
        velocity_key, time_key = jax.random.split(key)
        n_nodes = len(pos)
        state.velocity = np.where(
            redraw[:, None], sample_velocities(velocity_key, n_nodes, state.speed_min, state.speed_max), state.velocity
        )
        times = sample_segment_times(time_key, n_nodes, state.segment_min_s, state.segment_max_s)
        remaining_time = np.where(redraw, times, remaining_time)

    state.remaining_time = remaining_time
    state.pos = pos


def add_stations(key, state, n_new):
    """
    New stations at random positions within the bounds.
    """

    pos_key, velocity_key, time_key = jax.random.split(key, 3)
    pos = jax.random.uniform(pos_key, (n_new, 2), minval=state.bounds_min, maxval=state.bounds_max)
    velocity = sample_velocities(velocity_key, n_new, state.speed_min, state.speed_max)
    state.pos = np.concatenate([state.pos, np.asarray(pos, dtype=float)])
    state.velocity = np.concatenate([state.velocity, velocity])
    state.remaining_time = np.concatenate([
        state.remaining_time, sample_segment_times(time_key, n_new, state.segment_min_s, state.segment_max_s)
    ])
    state.mobile_mask = np.concatenate([state.mobile_mask, np.ones(n_new, dtype=bool)])


def nearest_ap_associations(pos, ap_indices):
    associations = {ap: [] for ap in ap_indices}

    for node in range(len(pos)):
        if node not in ap_indices:
            associations[ap_indices[np.argmin(((pos[ap_indices] - pos[node]) ** 2).sum(axis=1))]].append(node)

    return associations


class MobilityEnv:
    """
    Stations move within per-node bounds; APs and associations are fixed.
    """

    def __init__(self, scenario, bounds_min, bounds_max, mobile_mask, sim_dt, speed_min, speed_max, segment_min_s,
                 segment_max_s):
        self.scenario = scenario
        self.bounds_min, self.bounds_max, self.mobile_mask = bounds_min, bounds_max, mobile_mask
        self.sim_dt = sim_dt
        self.mobility_params = (speed_min, speed_max, segment_min_s, segment_max_s)

    def reset(self, key):
        initial_pos = np.asarray(self.scenario.pos, dtype=float)
        self.bounds_min[~self.mobile_mask] = initial_pos[~self.mobile_mask]
        self.bounds_max[~self.mobile_mask] = initial_pos[~self.mobile_mask]
        key, mobility_key = jax.random.split(key)
        self.state = create_mobility_state(
            mobility_key, initial_pos, self.bounds_min, self.bounds_max, self.mobile_mask, *self.mobility_params
        )
        return key

    def step(self, key, step):
        key, config_key, mobility_key = jax.random.split(key, 3)
        step_mobility_state(mobility_key, self.state, self.sim_dt)
        return key, config_key, {}


class EnterpriseEnv:
    """
    Stations move freely, new stations join periodically and stations reassociate to the nearest AP.
    """

    def __init__(self, scenario, ap_indices, bounds_min, bounds_max, sim_dt, speed_min, speed_max, segment_min_s,
                 segment_max_s, spawn_interval_steps, spawn_count, no_mobility):
        self.scenario = scenario
        self.ap_indices = ap_indices
        self.bounds_min, self.bounds_max = bounds_min, bounds_max
        self.sim_dt = sim_dt
        self.mobility_params = (0., 0., 1., 1.) if no_mobility else (speed_min, speed_max, segment_min_s, segment_max_s)
        self.spawn_interval_steps, self.spawn_count = spawn_interval_steps, spawn_count
        self.no_mobility = no_mobility

    def update_scenario(self):
        n_nodes = len(self.state.pos)
        self.scenario.pos = jnp.asarray(self.state.pos)
        self.scenario.associations = nearest_ap_associations(self.state.pos, self.ap_indices)
        self.scenario.walls = jnp.zeros((n_nodes, n_nodes))
        self.scenario.tx_power = jnp.full(n_nodes, self.scenario.tx_power[0].item())

    def reset(self, key):
        initial_pos = np.asarray(self.scenario.pos, dtype=float)
        mobile_mask = ~np.isin(np.arange(len(initial_pos)), self.ap_indices)
        key, mobility_key = jax.random.split(key)
        self.state = create_mobility_state(
            mobility_key, initial_pos, self.bounds_min, self.bounds_max, mobile_mask, *self.mobility_params
        )
        self.update_scenario()
        self.prev_associations, self.last_spawn_step = None, None
        return key

    def step(self, key, step):
        key, config_key, mobility_key, spawn_key = jax.random.split(key, 4)

        if not self.no_mobility:
            step_mobility_state(mobility_key, self.state, self.sim_dt)

        spawned = self.spawn_count > 0 and self.spawn_interval_steps > 0 and (
            self.last_spawn_step is None or step - self.last_spawn_step >= self.spawn_interval_steps
        )

        if spawned:
            add_stations(spawn_key, self.state, self.spawn_count)
            self.last_spawn_step = step

        self.update_scenario()
        topology_changed = self.scenario.associations != self.prev_associations

        if topology_changed:
            self.prev_associations = {ap: list(stas) for ap, stas in self.scenario.associations.items()}

        return key, config_key, {'spawned': spawned, 'topology_changed': topology_changed}


def add_env_args(args):
    args.add_argument('--env', type=str, choices=['residential', 'random', 'enterprise'], default='residential')
    args.add_argument('--scenario_seed', type=int, default=42)
    args.add_argument('--sim_time_s', type=float, default=20.)
    args.add_argument('--n_steps', type=int, default=None)
    args.add_argument('--sim_dt', type=float, default=TAU)
    args.add_argument('--mobility_speed_min', type=float, default=0.3)
    args.add_argument('--mobility_speed_max', type=float, default=1.5)
    args.add_argument('--mobility_segment_min_s', type=float, default=0.)
    args.add_argument('--mobility_segment_max_s', type=float, default=5.)
    args.add_argument('--n_sta_per_ap', type=int, default=4)
    args.add_argument('--room_size', type=float, default=None, help='Default: 10 (residential), 20 (enterprise)')
    args.add_argument('--no_walls', default=False, action='store_true', help='Residential only')
    args.add_argument('--d_ap', type=float, default=75., help='Random only')
    args.add_argument('--d_sta', type=float, default=4., help='Random only')
    args.add_argument('--n_ap', type=int, default=8, help='Random only')
    args.add_argument('--no_mobility', default=False, action='store_true', help='Enterprise only')
    args.add_argument('--spawn_interval_s', type=float, default=4., help='Enterprise only')
    args.add_argument('--spawn_count', type=int, default=10, help='Enterprise only')


def create_env(args):
    """
    Mobility environment and number of steps: residential (4 x 4 rooms), random (free space) or enterprise (open
    3 x 3 rooms with joining stations and reassociation).
    """

    if args.sim_dt <= 0.:
        raise ValueError('--sim_dt must be > 0')
    if not args.no_mobility and not 0. < args.mobility_speed_min <= args.mobility_speed_max:
        raise ValueError('Mobility speeds must satisfy 0 < min <= max')
    if not args.no_mobility and not 0. <= args.mobility_segment_min_s <= args.mobility_segment_max_s > 0.:
        raise ValueError('Mobility segment times must satisfy 0 <= min <= max, max > 0')

    n_steps = args.n_steps if args.n_steps is not None else max(1, int(round(args.sim_time_s / args.sim_dt)))
    mobility = (
        args.mobility_speed_min, args.mobility_speed_max, args.mobility_segment_min_s, args.mobility_segment_max_s
    )

    if args.env == 'residential':
        size = 10. if args.room_size is None else args.room_size
        scenario = residential_scenario(
            seed=args.scenario_seed, x_apartments=4, y_apartments=4, n_sta_per_ap=args.n_sta_per_ap, size=size,
            n_steps=n_steps, channel_width=80
        )

        if args.no_walls:
            scenario.walls = jnp.zeros_like(scenario.walls)

        return MobilityEnv(scenario, *room_bounds(4, 4, args.n_sta_per_ap, size), args.sim_dt, *mobility), n_steps

    if args.env == 'random':
        scenario = random_scenario(
            seed=args.scenario_seed, d_ap=args.d_ap, n_ap=args.n_ap, d_sta=args.d_sta, n_sta_per_ap=args.n_sta_per_ap,
            n_steps=n_steps, randomize=False, channel_width=80
        )
        return MobilityEnv(scenario, *area_bounds(scenario, args.d_ap), args.sim_dt, *mobility), n_steps

    size = 20. if args.room_size is None else args.room_size
    scenario = open_enterprise_scenario(
        seed=args.scenario_seed, x_apartments=3, y_apartments=3, n_sta_per_ap=args.n_sta_per_ap, size=size,
        n_steps=n_steps, channel_width=80
    )
    scenario.walls = jnp.zeros_like(scenario.walls)
    env = EnterpriseEnv(
        scenario, [room * (args.n_sta_per_ap + 1) for room in range(9)], np.zeros(2), np.full(2, 3 * size),
        args.sim_dt, *mobility, max(1, int(round(args.spawn_interval_s / args.sim_dt))), args.spawn_count,
        args.no_mobility
    )
    return env, n_steps
