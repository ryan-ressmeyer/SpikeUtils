"""
Interval jitter for spike resampling (Amarasingham et al. 2012).

Interval jitter is a conditional inference method for testing hypotheses
about the temporal resolution of neuronal spike trains. The central question
is whether observed spike timing reflects fine-timescale structure (e.g.,
synchrony, synaptic coupling) or can be explained by slower modulations of
firing rate alone.

The method is an instance of conditional modeling: inference is conditioned
on the spike counts within non-overlapping intervals of length delta. Given
that a spike exists in a particular interval, the null hypothesis asserts
that all placements within that interval are equally probable. This tests
whether temporal precision finer than delta is present in the data, without
requiring estimation of firing rates or trial-to-trial variability.

To generate a surrogate spike train, time is partitioned into contiguous
intervals of length delta. Each spike is independently and uniformly
repositioned within its interval, preserving the number of spikes per
interval (and hence any rate modulation on timescales coarser than delta)
while destroying temporal structure finer than delta. Repeating this
procedure produces an ensemble of surrogate datasets against which any
test statistic (e.g., synchrony count, cross-correlogram shape, mutual
information) can be compared to form acceptance bands or compute z-scores.

Unlike trial shuffling, interval jitter does not assume statistical
repeatability across trials and applies equally to single-trial data. It
provides an exact hypothesis test of a large class of null hypotheses for
spike-generating processes conditioned on interval spike counts
(Amarasingham et al. 2012, Sec. 2.2). The method is recommended over trial
shuffling when the goal is to isolate fine-temporal correlations from slow
co-modulation of firing rates (Amarasingham et al. 2012, Sec. 2.7).

Choice of delta:
  The interval width delta defines the temporal resolution being tested.
  It should exceed the timescale of the interaction of interest (so that
  jitter can disrupt it) but remain small enough to preserve rate
  co-modulation. Platkiewicz et al. (2017) show that overly large delta
  can introduce artifacts when rate functions vary substantially within
  a single interval. Amarasingham et al. (2012, Sec. 3.1) provide exact
  tests for bounding the rate of change of firing probabilities within
  intervals, enabling principled selection of delta.

References:
  - Amarasingham, Harrison, Hatsopoulos & Geman (2012). Conditional modeling
    and the jitter method of spike resampling. J Neurophysiol 107(2):517-536.
    doi:10.1152/jn.00633.2011
  - Platkiewicz, Stark & Amarasingham (2017). Spike-centered jitter can
    mistake temporal structure. Neural Computation 29(3):783-803.
  - Fujisawa, Amarasingham, Harrison & Buzsaki (2008). Behavior-dependent
    short-term assembly dynamics in the medial prefrontal cortex. Nature
    Neuroscience 11(7):823-833.
  - Amarasingham, Harrison & Bhagwat (2011). Technical appendix to
    "Conditional modeling and the jitter method of spike resampling."
    Brown University technical report.
"""

import numpy as np


def interval_jitter_spikes(spike_times, spike_clusters, delta, rng):
    """
    Generate a surrogate spike dataset via interval jitter.

    Time is partitioned into contiguous, non-overlapping intervals of width
    delta. Each spike is independently and uniformly repositioned within the
    interval it belongs to. The spike count per interval per cluster is
    exactly preserved, so any rate modulation on timescales coarser than
    delta is retained in the surrogate. Temporal structure finer than delta
    is destroyed.

    This implements the Monte Carlo interval jitter procedure described in
    Amarasingham et al. (2012, Sec. 2.1, Fig. 2A).

    The implementation is vectorized across all spikes regardless of cluster
    identity, since the jitter operation (floor + uniform reposition) is
    identical per spike.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Spike times in seconds, sorted in ascending order.
    spike_clusters : ndarray, shape (n_spikes,)
        Cluster/unit ID for each spike.
    delta : float
        Jitter interval width in seconds. Defines the temporal resolution
        being tested: the null hypothesis is that all intraintreval spike
        placements are equally likely, i.e., there is no temporal precision
        finer than delta. Must be positive.
    rng : numpy.random.Generator
        NumPy random number generator instance (e.g., from
        ``np.random.default_rng(seed)``).

    Returns
    -------
    jittered_times : ndarray, shape (n_spikes,)
        Jittered spike times, sorted in ascending order.
    jittered_clusters : ndarray, shape (n_spikes,)
        Cluster IDs corresponding to jittered_times (reordered to match
        the ascending sort of jittered times).
    """
    interval_idx = np.floor(spike_times / delta)
    jittered = (interval_idx + rng.uniform(size=len(spike_times))) * delta
    sort_idx = np.argsort(jittered)
    return jittered[sort_idx], spike_clusters[sort_idx]
