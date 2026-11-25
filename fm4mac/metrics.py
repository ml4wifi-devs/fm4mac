import jax.numpy as jnp
import numpy as np


def pf_utility(thr, floor=1.):
    """
    Proportional-fairness utility: sum of the logarithms of per-device throughputs (below `floor` count as `floor`).
    """

    return jnp.log(jnp.maximum(thr, floor)).sum(axis=-1)


def gm_ratio(utility, reference, n_devices):
    """
    Ratio of the geometric-mean throughputs of a method and a reference given their PF utilities; does not depend
    on the number of devices.
    """

    return np.exp((np.asarray(utility) - np.asarray(reference)) / n_devices)


def jain(thr):
    return thr.sum(axis=-1) ** 2 / (thr.shape[-1] * (thr ** 2).sum(axis=-1))


def device_metrics(thr, floor=1.):
    """
    Mean over configurations of the PF utility, sum, minimum, 10th percentile, Jain's index, and number of starved
    devices of per-device throughputs `[n, n_devices]`.
    """

    return dict(
        pf=float(pf_utility(thr, floor).mean()),
        sum=float(thr.sum(-1).mean()),
        min=float(thr.min(-1).mean()),
        p10=float(np.percentile(thr, 10, axis=-1).mean()),
        jain=float(jain(thr).mean()),
        starved=float((thr < floor).sum(-1).mean())
    )


def agreement(reference, value, reference_se, value_se, threshold=10.):
    """
    Agreement between a reference (e.g. Komondor) and another simulator: Pearson correlation, absolute and relative
    errors (relative ones only above `threshold`), and the fraction of values within noise.
    """

    err = value - reference
    big = reference > threshold
    ape = np.abs(err[big]) / reference[big]
    z = err / np.sqrt(reference_se ** 2 + value_se ** 2 + 1e-9)
    return dict(
        n=int(len(reference)),
        pearson=float(np.corrcoef(reference, value)[0, 1]),
        mae=float(np.abs(err).mean()),
        bias=float(err.mean()),
        mape=float(ape.mean()),
        p90_ape=float(np.percentile(ape, 90)),
        within_5pct_or_2mbps=float(np.mean((np.abs(err) <= 0.05 * reference) | (np.abs(err) <= 2.))),
        within_2se=float(np.mean(np.abs(z) <= 2.))
    )
