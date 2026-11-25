import jax
import jax.numpy as jnp
import numpy as np


def create_coordinate_ascent(score_fn):
    """
    Coordinate ascent over the devices of many configurations at once. A configuration is a vector of per-device
    decisions; device `d` may take any value `options[d, o]` with `valid[..., d, o]`. Each device (random order per
    configuration and round) takes the option that maximises `score_fn(args, candidates[n, n_options, n_devices])
    -> [n, n_options]` with the other devices fixed. With a surrogate as the scorer this is refinement; with a
    simulator, local search.
    """

    @jax.jit
    def step(args, decisions, options, valid, device):
        rows = jnp.arange(decisions.shape[0])
        values = options[device]
        at_device = jax.nn.one_hot(device, decisions.shape[1], dtype=bool)[:, None]
        candidates = jnp.where(at_device, values[..., None], decisions[:, None])
        scores = jnp.where(valid[rows, device], score_fn(args, candidates), -jnp.inf)
        best = jnp.argmax(scores, axis=-1)
        return candidates[rows, best], scores[rows, best], candidates, scores

    def run(args, decisions, options, valid, rng, n_rounds=2, visited=False):
        """
        decisions: [n, n_devices] initial configurations; options: [n_devices, n_options]; valid: boolean, broadcast
        to [n, n_devices, n_options]. Stops early when a round changes nothing. Returns the final configurations and
        their scores, and with `visited` also every evaluated (candidates, scores) pair.
        """

        decisions, options = jnp.asarray(decisions), jnp.asarray(options)
        valid = jnp.broadcast_to(jnp.asarray(valid), decisions.shape + options.shape[-1:])
        order = np.tile(np.arange(decisions.shape[1]), (decisions.shape[0], 1))
        scores, history = None, []

        for _ in range(n_rounds):
            start = decisions

            for device in rng.permuted(order, axis=1).T:
                decisions, scores, candidates, candidate_scores = step(
                    args, decisions, options, valid, jnp.asarray(device)
                )

                if visited:
                    history.append((np.asarray(candidates), np.asarray(candidate_scores)))

            if jnp.array_equal(start, decisions):
                break

        out = np.asarray(decisions), np.asarray(scores)
        return out + (history,) if visited else out

    return run


def select_top(scores, top_k):
    return np.argsort(-scores)[:top_k]


def select_covering(scores, covers, top_k, coverage_ratio=1.):
    """
    Top-k by score in which the first `ceil(top_k * coverage_ratio)` picks must each cover something new (e.g. serve
    a station no earlier pick serves); `covers` is a boolean `[n, n_items]` matrix.
    """

    order = np.argsort(-scores)
    covered = np.zeros(covers.shape[-1], dtype=bool)
    n_covering = int(np.ceil(top_k * coverage_ratio))
    chosen = []

    for i in order:
        if len(chosen) < n_covering and (covers[i] & ~covered).any():
            chosen.append(i)
            covered |= covers[i]

    chosen += [i for i in order if i not in chosen][:top_k - len(chosen)]
    return np.asarray(chosen)


def rank_correlation(a, b):
    """
    Mean Spearman correlation between the rows of `a` and `b`.
    """

    ranks_a, ranks_b = a.argsort(axis=-1).argsort(axis=-1), b.argsort(axis=-1).argsort(axis=-1)
    return np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(ranks_a, ranks_b)]).item()
