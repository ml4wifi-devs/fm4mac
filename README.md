# Teaching Generative Models to Configure Wi-Fi

fm4mac is a JAX library for configuring Wi-Fi networks with generative models. It covers combinatorial medium access control (MAC) problems such as channel allocation, station association and coordinated spatial reuse (Co-SR). A GNN encodes the network, flow matching samples candidate configurations, a learned surrogate selects and refines them, and RL adapts the pipeline to a particular deployment. It generalises [FM4WiFi](https://github.com/ml4wifi-devs/lai4wifi) from Co-SR to any MAC decision that can be attached to a network graph and evaluated in a simulator.

The examples build on the following repositories:

- [mapc-sim](https://github.com/ml4wifi-devs/mapc-sim): Co-SR simulator (SINR, MCS and data rates of parallel transmissions)
- [mapc-optimal](https://github.com/ml4wifi-devs/mapc-optimal): F-/T-Optimal Co-SR schedules by column generation
- [mapc-optimal-research](https://github.com/ml4wifi-devs/mapc-optimal-research): Co-SR scenarios (residential, enterprise, random)
- [mapc-mab](https://github.com/ml4wifi-devs/mapc-mab): hierarchical multi-armed bandits (H-MAB) for Co-SR
- [mapc-dcf](https://github.com/ml4wifi-devs/mapc-dcf): DCF baseline for Co-SR

## Pipeline

```
network graph + observation ──► Encode (GNN) ──► Generate (flow matching) ──► Decode ──► Score (surrogate)
                                                         ▲                                     │
                                                         │ RL (DiffusionNFT, online mode)      ▼
                                                 network / simulator ◄── Apply ◄── Refine (surrogate-guided search)
```

1. **Encode.** A network is a graph whose nodes are devices and whose edges carry measurements. Every decision lives on an edge, and per-device decisions live on self-loops. A variational GNN autoencoder maps every edge, including its decision, to a latent vector.
2. **Generate and decode.** Conditional flow matching samples latents of good configurations given the latents of one observed configuration (the context). The decoder turns them into candidate configurations.
3. **Score.** A surrogate GNN reads the explicit configuration graph and predicts a per-device metric with a Gaussian head. The predicted objective is the sum of the predictions over the decision edges, so errors stay local and the model transfers to larger networks. It is trained with a Gaussian NLL plus a pairwise ranking loss within groups of configurations of the same network.
4. **Refine.** Coordinate ascent over the devices: each device takes the option with the best predicted objective while the others stay fixed. With a simulator as the scorer, the same function is the local-search reference.
5. **Adapt (online mode).** DiffusionNFT moves the generator toward samples that beat their group. Rewards can be given per device (neighbourhood credit) instead of per network, and the surrogate is retrained on the same evaluations.

The pipeline runs in two modes. In zero-shot mode, pretrained models configure a new network without any trials in it. In online mode, they are first fine-tuned with RL on the deployment.

## Install

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                    # library (CPU)
uv sync --extra gpu                        # + CUDA 12
uv sync --extra examples                   # + channel allocation and association examples
uv sync --extra cosr                       # + Co-SR example (mapc-* packages)
uv sync --extra gpu --extra cosr --group dev   # everything, incl. ruff
```

Every entry point sets `JAX_ENABLE_X64=True` before importing JAX: random deployments and samples depend on it. GPU runs are not bit-reproducible, because graph aggregations use scatter-add.

## Layout

```
fm4mac/
├── fm4mac/
│   ├── graphs.py        # JaxEnum edge attributes, batching/padding, graph utilities
│   ├── nn.py            # MLP, segment-masked transformer, graph network layers
│   ├── autoencoder.py   # GNN autoencoder with heads from an edge-attribute dataclass, VAE losses
│   ├── flow.py          # conditional flow matching, Euler sampling, FM loss, conditioning input, MMD
│   ├── surrogate.py     # explicit-configuration surrogate, NLL + ranking loss, utility
│   ├── search.py        # coordinate ascent (refinement / local search), top-k and covering selection
│   ├── rl.py            # DiffusionNFT loss, group advantages, neighbourhood credit, old-model update
│   ├── training.py      # optimiser, jitted train step, generic training loop
│   ├── data.py          # dataset I/O, prebatching by edge budget, loaders
│   ├── checkpoints.py   # orbax checkpoints, configs, JSON-lines logger
│   └── metrics.py       # proportional fairness, geometric-mean ratio, Jain, simulator agreement
├── examples/
│   ├── wlan_sim/        # JAX port of Komondor v4.2.1 (CSMA/CA, dynamic channel bonding) + validation
│   ├── channel/         # channel allocation with dynamic channel bonding
│   ├── association/     # AP-station association
│   └── cosr/            # coordinated spatial reuse (multi-AP coordination, IEEE 802.11bn)
└── tests/               # unit tests of the library
```

## Applying fm4mac to a new MAC problem

A problem supplies the graph, the decisions and the evaluation, and the library supplies the rest:

```python
import os
os.environ['JAX_ENABLE_X64'] = 'True'

import jax.numpy as jnp

from fm4mac.autoencoder import Autoencoder, mixture_categorical_loss, squared_error
from fm4mac.graphs import JaxEnum, edge_attributes
from fm4mac.search import create_coordinate_ascent
from fm4mac.surrogate import Surrogate


class Channel(JaxEnum):        # a categorical decision; the last class is "not applicable"
    CH_1 = 0
    CH_2 = 1
    NA = 2


@edge_attributes
class Edge:                    # one field per edge attribute; JaxEnum fields are categorical
    channel: Channel
    rssi: float


autoencoder = Autoencoder(Edge, embed_features=32, encoder_features=(64, 64), decoder_features=(64, 64),
                          latent_features=32, n_layers=3, dtype=jnp.float32)
field_losses = {'channel': mixture_categorical_loss, 'rssi': squared_error}   # for autoencoder_loss
surrogate = Surrogate(gnn_layers=3, dim=64, ff_dim=128, n_layers=1, n_heads=4, drop_rate=0.1,
                      dtype=jnp.float32, embed_nodes=False)


def score_fn(params, candidates):
    """
    Predicted objective of candidate configurations [n, n_options, n_devices] -> [n, n_options], e.g. the surrogate
    applied to their configuration graphs, or a simulator for local search.
    """
    ...


refine = create_coordinate_ascent(score_fn)
# configurations, scores = refine(params, initial, options, valid, rng, n_rounds=2)
```

The remaining parts compose the same way:

- `fm4mac.flow.FlowMatching`, `flow_matching_loss` and `fm4mac.training.fit` train the generator on (context, target) latent pairs.
- `fm4mac.surrogate.surrogate_loss` trains the surrogate.
- `fm4mac.rl.create_nft_loss`, `group_advantages`, `nft_rewards` and `local_utilities` implement the RL step.

The examples are complete instances of this recipe.

## Examples

Run everything from the repository root. Paths are relative to the example directory, and each example ships the checkpoints used in the tutorial in `examples/<name>/runs/`.

### Wi-Fi simulator (`examples/wlan_sim`)

`examples/wlan_sim` is an event-driven CSMA/CA simulator with dynamic channel bonding (OP, SCB, always-max), RTS/CTS, A-MPDU aggregation and the 802.11ax CCA thresholds. It reproduces the behaviour of [Komondor](https://github.com/wn-upf/Komondor) v4.2.1, including its RTS/CTS dynamic-bandwidth rule, and simulates thousands of networks in parallel on a GPU (`jax.vmap`). Per-station counters and a slot-to-node mapping support association.

```bash
python -m examples.wlan_sim.validate channel        # 200 random scenarios, 3 Komondor seeds, 10 JAX seeds
python -m examples.wlan_sim.validate association
python -m examples.wlan_sim.analyze --input examples/wlan_sim/validation/channel.json
python -m unittest examples.wlan_sim.test_wlan_sim  # incl. live Komondor regression cases if KOMONDOR_DIR is set
```

Agreement with Komondor v4.2.1 (`examples/wlan_sim/validation/`, devices above 10 Mb/s):

| | Pearson | Mean abs. error | 90th percentile |
|---|---|---|---|
| Channel plans, per WLAN | 0.9996 | 1.4% | 3.5% |
| Channel plans, aggregate | 0.9999 (Spearman) | 0.4% | |
| Association, per station | 0.9995 | 3.0% | 6.4% |
| Association, PF utility | 0.9999 | | |

### Channel allocation (`examples/channel`)

Each AP selects a primary 20 MHz channel and a maximum bandwidth (32 options) to maximise proportional fairness (PF) under dynamic channel bonding. The plan lives on AP self-loops, and RL rewards each AP by the PF of its interference neighbourhood.

```bash
python -m examples.channel.dataset --objective pf --seed 11 --n_deployments 300 --output datasets/pf_train_11.pkl.lz4
python -m examples.channel.train_autoencoder                      # -> runs/autoencoder
python -m examples.channel.train_flow_matching                    # -> runs/flow_matching
python -m examples.channel.train_surrogate                        # -> runs/surrogate
python -m examples.channel.rl name=rl_8 deployments=[random/8]    # online mode (rl_preset=medium|large for 32/48 APs)
python -m examples.channel.baselines --deployment random/8        # local search, colouring, bandits
python -m examples.channel.evaluate --deployment random/8         # zero-shot + refinement
python -m examples.channel.evaluate --deployment random/8 --run runs/rl_8              # online mode + refinement
python -m examples.channel.evaluate --deployment random/8 --init heuristic             # surrogate search from heuristics
```

The tutorial also used datasets with `--objective sum` (seeds 1–6, validation seed 100) for the autoencoder and surrogate. Flow matching training with `max_edges=4096` needs about 16 GB of GPU memory.

### Station association (`examples/association`)

Each station selects one of its candidate APs to maximise PF over stations. The decision lives on candidate AP-station edges, and RL rewards each station by the PF of the APs it can affect. The baselines are strongest signal, estimated throughput, an airtime model and per-station bandits.

```bash
python -m examples.association.dataset --seed 101 --n_deployments 30 --output datasets/train_101.pkl.lz4
python -m examples.association.train_autoencoder
python -m examples.association.train_flow_matching
python -m examples.association.train_surrogate
python -m examples.association.rl name=rl_8 deployment=random/8   # rl_preset=large32|large48 for 32/48 APs
python -m examples.association.baselines --deployment random/8
python -m examples.association.evaluate --deployment random/8
python -m examples.association.evaluate --deployment random/8 --run runs/rl_8 --label rl
```

### Coordinated spatial reuse (`examples/cosr`)

In each transmission opportunity, every AP is either off or serves one of its stations with one of 4 power levels and 14 MCSs. Eight configurations form a schedule, and each configuration serves different designated stations. Evaluation uses [mapc-sim](https://github.com/ml4wifi-devs/mapc-sim) on residential buildings of 4–30 APs.

```bash
python -m examples.cosr.dataset --generator random --n_configurations 30 --output datasets/random.pkl.lz4  # also h_mab, f_optimal, t_optimal (CPLEX or --solver cbc)
python -m examples.cosr.train_autoencoder
python -m examples.cosr.train_flow_matching
python -m examples.cosr.train_surrogate
python -m examples.cosr.rl                                                               # online mode, residential/3 (rl_preset=residential_<i> for others)
python -m examples.cosr.evaluate --scenario residential/3 --n_designated 3               # zero-shot (top-8 with and without refinement)
python -m examples.cosr.evaluate --scenario residential/3 --n_designated 3 --run runs/rl_residential_3
python -m examples.cosr.evaluate --scenario residential/3 --candidates random            # ablations: --scores random, --no_refine
python -m examples.cosr.baselines --method sr_groups --idx 3                             # also t_optimal, f_optimal, h_mab, dcf
python -m examples.cosr.dynamic --method zero_shot --env residential --output results/dynamic/zs.npz  # mobility, joins
```

Only the RL checkpoints for residential/3, /6 and /7 are shipped.

### Reproducing the tutorial

With the shipped checkpoints, the examples reproduce the tutorial results. For Co-SR at 12 APs (top-8 throughput relative to T-Optimal):

| Zero-shot | Zero-shot + refinement | Online + refinement |
|---|---|---|
| 0.375 | 0.711 | 0.765 |

The channel and association results at 8 APs match within the run-to-run variation of the local-search reference.

## Tests

```bash
uv run python -m unittest discover -s tests -t .                         # library
uv run python -m unittest discover -s examples -p "test_*.py" -t .       # examples
uv run ruff check fm4mac tests examples
```

CI (`.github/workflows/ci.yml`) runs the same commands.

## Citation

If you use fm4mac in your research, please cite:

```bibtex
@article{wojnar2026teaching,
  title   = {Teaching Generative Models to Configure {Wi-Fi}},
  author  = {Wojnar, Maksymilian},
  year    = {2026},
  note    = {Under review}
}
```
