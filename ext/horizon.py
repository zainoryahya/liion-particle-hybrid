"""Sample times used for the sharp l_inf gain (Theorem 1).

The gain G_k = |delta_0| + sum_{j<=k} |delta_j - delta_{j-1}| is non-decreasing in k, and the l_inf-induced norm of
the error map is its limit as k -> infinity. We evaluate it over a fixed PHYSICAL horizon of T_H = 5 dimensionless
diffusion times (tau = D t / R^2), after which every error considered here has reached its constant long-time value
(the slowest particle mode decays like exp(-20.19 tau)).

All samples are integer multiples of dtau. The first DENSE samples are taken consecutively; beyond them a
geometric subset of the sample indices is used. On any stretch where the error is monotone, this gives exactly the
same total variation as using every sample, so the result equals the full-sample gain whenever the error has no
extremum between retained samples (checked in gain_check.py).
"""
import numpy as np

T_H = 5.0
DENSE = 2000
NLOG = 4000


def sample_index(dtau, T=T_H, dense=DENSE, nlog=NLOG):
    kmax = int(np.ceil(T / dtau))
    k = np.arange(min(dense, kmax) + 1)
    if kmax > dense:
        k = np.unique(np.r_[k, np.round(np.geomspace(dense, kmax, nlog)).astype(np.int64)])
    return k


def lags_for(dtau, **kw):
    return sample_index(dtau, **kw) * dtau


def gain(delta):
    delta = np.asarray(delta)
    return float(abs(delta[0]) + np.abs(np.diff(delta)).sum())
