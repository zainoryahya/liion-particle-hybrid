"""Sample times used for the sharp l_inf gain (Theorem 1).

The gain G_k = |delta_0| + sum_{j<=k} |delta_j - delta_{j-1}| is non-decreasing in k, and the l_inf-induced norm of
the error map is its limit G as k -> infinity.

Samples up to T_H = 5 dimensionless diffusion times (tau = D t / R^2): the first DENSE samples consecutively, beyond
them a geometric subset of NLOG sample indices. On any stretch where the error is monotone this gives exactly the same
total variation as using every sample (checked in gain_check.py). By tau = 5 the exact and the FV/transfer-function
step responses have settled (slowest particle mode exp(-20.19 tau)), but the correction states of the hybrid schemes
decay more slowly (per-mesh rates >= 0.1). Up to v1.1.0 the gain was evaluated only to T_H, which underestimates the
gain of the corrected schemes (by up to 4 % for the per-mesh correction and 10 % for the universal one).

Since v1.1.1 the samples continue geometrically (NTAIL sample-aligned times) from T_H to T_TAIL = 2000, by which every
per-mesh correction state has decayed (exp(-200)). The universal correction has rates down to 1e-6/h^2; its gain is
completed in hybrid_theory2.py with a closed-form tail (the FV and exact parts are constant beyond T_H), see
gain_with_tail().
"""
import numpy as np

T_H = 5.0
DENSE = 2000
NLOG = 4000
T_TAIL = 2000.0
NTAIL = 1500


def sample_index(dtau, T=T_H, dense=DENSE, nlog=NLOG):
    kmax = int(np.ceil(T / dtau))
    k = np.arange(min(dense, kmax) + 1)
    if kmax > dense:
        k = np.unique(np.r_[k, np.round(np.geomspace(dense, kmax, nlog)).astype(np.int64)])
    return k


def tail_index(dtau, T0=T_H, T1=T_TAIL, n=NTAIL):
    k0 = int(np.ceil(T0 / dtau)); k1 = int(np.ceil(T1 / dtau))
    k = np.unique(np.round(np.geomspace(k0, k1, n)).astype(np.int64))
    return k[k > k0]


def lags_for(dtau, tail=True, ntail=NTAIL, **kw):
    k = sample_index(dtau, **kw)
    if tail:
        k = np.r_[k, tail_index(dtau, n=ntail)]
    return k * dtau


def gain_with_tail(delta_H, delta_far):
    """gain over the sampled horizon plus the variation of a far tail (delta_far starts after the last sample of
    delta_H; every far sample is a multiple of dtau, sampled densely enough that the tail has no missed extremum)."""
    delta_H = np.asarray(delta_H); delta_far = np.asarray(delta_far)
    return gain(delta_H) + float(abs(delta_far[0] - delta_H[-1]) + np.abs(np.diff(delta_far)).sum())


def gain(delta):
    delta = np.asarray(delta)
    return float(abs(delta[0]) + np.abs(np.diff(delta)).sum())
