from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm


def _scatter_add(flat_arr, indices, xp):
    """Add 1 at each index, handling duplicate indices correctly."""
    if xp is np:
        np.add.at(flat_arr, indices, 1)
    else:
        import cupyx
        cupyx.scatter_add(flat_arr, indices, 1)


def calc_ccgs(
    spike_times: np.ndarray,
    bin_edges: np.ndarray,
    spike_clusters: Optional[np.ndarray] = None,
    uids: Optional[np.ndarray] = None,
    progress: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute all pairwise cross-correlograms among a set of spike clusters.

    This function uses a highly efficient, vectorized algorithm that iterates
    by "shift" in the spike array rather than by individual spike pairs. It
    skips counting spike pairs with a time lag of zero, meaning the central
    bin of an autocorrelogram will not equal the total spike count.

    When ``device="cuda"``, the main loop runs on GPU via CuPy using the
    identical algorithm — CuPy's NumPy-compatible API means the same code
    path handles both backends.

    Originally inspired by phylib correlogram function.
    https://github.com/cortex-lab/phylib/blob/master/phylib/stats/ccg.py

    Parameters
    ----------
    spike_times : np.ndarray (n_spikes,)
        A 1D array of spike times in seconds. For best performance, these
        should be pre-sorted, though the function will sort them if needed.
    bin_edges : np.ndarray (n_bins + 1,)
        A 1D array defining the edges of the time lag bins, in seconds.
        Example: `np.linspace(-0.05, 0.05, 101)` creates 100 bins for a
        +/- 50 ms window.
    spike_clusters : np.ndarray (n_spikes,), optional
        A 1D integer array mapping each spike to a cluster ID. If None,
        all spikes are treated as belonging to a single cluster.
    uids : np.ndarray (n_clusters,), optional
        An ordered 1D array of the unique cluster IDs to include in the
        output. The order of this array determines the axes of the output
        matrix. Spikes belonging to clusters not in `uids` will be ignored.
        If None, all unique clusters from `spike_clusters` will be used,
        sorted in ascending order.
    progress : bool, default=False
        If True, display a `tqdm` progress bar during computation.
    device : str, default="cpu"
        ``"cpu"`` for NumPy, ``"cuda"`` for CuPy GPU acceleration.

    Returns
    -------
    correlograms : np.ndarray
        A 3D array of shape `(n_clusters, n_clusters, n_bins)` containing
        integer counts. `correlograms[i, j, k]` is the number of spikes from
        cluster `uids[j]` that occurred in time bin `k` relative to a spike
        from cluster `uids[i]`.
    """
    # --- 1. Input Validation and Preparation (always CPU / NumPy) ---
    spike_times = np.asarray(spike_times, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    assert spike_times.ndim == 1, "spike_times must be a 1D array."
    assert bin_edges.ndim == 1, "bin_edges must be a 1D array."
    assert np.all(np.diff(bin_edges) > 0), "Bin edges must be monotonically increasing."

    if spike_clusters is None:
        spike_clusters = np.zeros(len(spike_times), dtype=np.int32)
    else:
        spike_clusters = np.asarray(spike_clusters, dtype=np.int32)

    assert len(spike_times) == len(spike_clusters), \
        "Spike times and spike clusters must have the same length."

    if not np.all(np.diff(spike_times) >= 0):
        print("Spike times are not sorted, sorting...")
        sort_inds = np.argsort(spike_times)
        spike_times = spike_times[sort_inds]
        spike_clusters = spike_clusters[sort_inds]

    # --- 2. Cluster ID Filtering and Mapping (CPU) ---
    if uids is None:
        uids = np.unique(spike_clusters)
        needs_reordering = False
    else:
        uids = np.asarray(uids, dtype=np.int32)
        needs_reordering = True

    membership_mask = np.isin(spike_clusters, uids)
    spike_times = spike_times[membership_mask]
    spike_clusters = spike_clusters[membership_mask]

    n_clusters = len(uids)
    unsort_indices = None

    if needs_reordering:
        sort_indices = np.argsort(uids)
        sorted_uids = uids[sort_indices]
        unsort_indices = np.argsort(sort_indices)
    else:
        sorted_uids = uids

    spike_inds = np.searchsorted(sorted_uids, spike_clusters).astype(np.int32)

    # --- 3. Select array backend ---
    if device == "cpu":
        xp = np
    else:
        import cupy as cp
        xp = cp
        spike_times = cp.asarray(spike_times)
        spike_inds = cp.asarray(spike_inds)

    n_spikes = len(spike_times)
    n_bins = len(bin_edges) - 1
    ccgs = xp.zeros((n_clusters, n_clusters, n_bins), dtype=xp.int32)
    min_bin, max_bin = bin_edges[0], bin_edges[-1]

    # Optimization: fast path for linearly spaced bins
    bin_diffs = np.diff(bin_edges)
    if np.allclose(bin_diffs, bin_diffs[0]):
        bin_width = bin_diffs[0]
        def digitize(x):
            bins = (x - min_bin) / bin_width
            return xp.clip(bins.astype(xp.int32), 0, n_bins - 1)
    else:
        xp_bin_edges = xp.asarray(bin_edges)
        def digitize(x):
            return xp.searchsorted(xp_bin_edges, x).astype(xp.int32) - 1

    # --- 4. Main Correlogram Loop ---
    shift = 1
    pos_mask = xp.ones(n_spikes, dtype=bool)
    neg_mask = xp.ones(n_spikes, dtype=bool)
    pbar = tqdm(total=1.0, desc="Calculating CCGs", disable=not progress)

    while True:
        # --- Positive Lags (Ref: i, Target: i + shift) ---
        pos_mask[-shift:] = False
        active_pos_indices = xp.where(pos_mask[:-shift])[0]
        has_pos = len(active_pos_indices) > 0

        if has_pos:
            pos_dts = spike_times[active_pos_indices + shift] - spike_times[active_pos_indices]
            valid_pos = (min_bin < pos_dts) & (pos_dts < max_bin)
            if xp.any(valid_pos):
                valid_indices = active_pos_indices[valid_pos]
                pos_i, pos_j = spike_inds[valid_indices], spike_inds[valid_indices + shift]
                pos_bins = digitize(pos_dts[valid_pos])
                ravel_inds = xp.ravel_multi_index((pos_i, pos_j, pos_bins), ccgs.shape)
                _scatter_add(ccgs.ravel(), ravel_inds, xp)
            pos_mask[active_pos_indices] = pos_dts < max_bin

        # --- Negative Lags (Ref: i + shift, Target: i) ---
        neg_mask[:shift] = False
        active_neg_indices = xp.where(neg_mask[shift:])[0]
        has_neg = len(active_neg_indices) > 0

        if has_neg:
            neg_dts = spike_times[active_neg_indices] - spike_times[active_neg_indices + shift]
            valid_neg = (min_bin < neg_dts) & (neg_dts < max_bin)
            if xp.any(valid_neg):
                valid_indices = active_neg_indices[valid_neg]
                neg_i, neg_j = spike_inds[valid_indices + shift], spike_inds[valid_indices]
                neg_bins = digitize(neg_dts[valid_neg])
                ravel_inds = xp.ravel_multi_index((neg_i, neg_j, neg_bins), ccgs.shape)
                _scatter_add(ccgs.ravel(), ravel_inds, xp)
            neg_mask[active_neg_indices + shift] = neg_dts > min_bin

        # --- Loop Control and Progress Update ---
        if not has_pos and not has_neg:
            pbar.n = 1.0
            pbar.set_description("Calculating CCGs: Done")
            break

        if progress:
            progress_val = 1.0 - float(xp.sum(pos_mask) + xp.sum(neg_mask)) / (2 * n_spikes)
            pbar.n = round(progress_val, 3)
            pbar.set_description(f"Calculating CCGs: Shift {shift}")

        shift += 1

    pbar.close()

    # --- 5. Transfer back to NumPy if needed ---
    if device != "cpu":
        ccgs = ccgs.get()

    if needs_reordering:
        ccgs = ccgs[unsort_indices][:, unsort_indices]

    return ccgs


def calc_local_firing_rate(spike_times: np.ndarray, fr_est_dur: float = 1.0) -> float:
    """
    Compute the local (autocorrelation-weighted) firing rate of a spike train.

    Produces the same result as
    ``calc_ccgs(spike_times, [0, fr_est_dur]).squeeze() / fr_est_dur / n_spikes``
    but entirely via vectorised NumPy, without the CCG histogram overhead.

    For each spike i, count the number of spikes j > i with
    ``spike_times[j] - spike_times[i] < fr_est_dur``.  Summing those counts
    and dividing by ``n_spikes * fr_est_dur`` gives the same ACG-density
    estimate, which correctly tracks the *local* firing rate:  a neuron active
    for only 10 min of a 2-hour recording returns its true ~20 Hz rate, not the
    global time-averaged ~1.6 Hz.

    Parameters
    ----------
    spike_times : np.ndarray (n_spikes,)
        Sorted spike times in seconds.
    fr_est_dur : float
        Half-window duration in seconds (default 1.0).

    Returns
    -------
    firing_rate : float
        Local firing rate in Hz.
    """
    n_spikes = len(spike_times)
    if n_spikes == 0:
        return 0.0
    window_ends = np.searchsorted(spike_times, spike_times + fr_est_dur, side='right')
    spikes_in_window = window_ends - np.arange(n_spikes) - 1
    return float(np.sum(spikes_in_window) / (n_spikes * fr_est_dur))


def calc_single_bin_ccgs(
    spike_times: np.ndarray,
    window: tuple[float, float],
    spike_clusters: Optional[np.ndarray] = None,
    uids: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute single-bin cross-correlograms for all cluster pairs.

    For each reference spike from cluster *i*, counts how many spikes from
    cluster *j* fall within the time lag window and accumulates the result
    into an ``(n_clusters, n_clusters)`` matrix.

    Streamlined shift-based algorithm optimised for the single-bin case:
    no bin digitisation, and only forward shifts are computed (negative
    windows are handled by mirroring internally).

    Note: ``result[i, j]`` with a positive-lag window ``(a, b)`` equals
    ``result.T[i, j]`` with the mirrored negative-lag window ``(-b, -a)``.
    So you only need one call to get both directions.

    Parameters
    ----------
    spike_times : np.ndarray (n_spikes,)
        Sorted spike times in seconds.
    window : tuple[float, float]
        ``(start, end)`` of the time lag window in seconds. ``start`` must be
        less than ``end``. Both positive and negative windows are supported.
    spike_clusters : np.ndarray (n_spikes,), optional
        Cluster ID per spike. If None, all spikes belong to one cluster.
    uids : np.ndarray (n_clusters,), optional
        Ordered unique cluster IDs for the output axes. Spikes from clusters
        not in ``uids`` are ignored. If None, uses all unique clusters.
    device : str, default="cpu"
        ``"cpu"`` for NumPy, ``"cuda"`` for CuPy GPU acceleration.

    Returns
    -------
    counts : np.ndarray (n_clusters, n_clusters), dtype int32
        ``counts[i, j]`` is the number of spikes from cluster ``uids[j]``
        that fell within the window relative to spikes from cluster ``uids[i]``.
    """
    # --- 1. Input validation (CPU) ---
    spike_times = np.asarray(spike_times, dtype=np.float64)
    assert spike_times.ndim == 1, "spike_times must be a 1D array."

    win_start, win_end = float(window[0]), float(window[1])
    assert win_start < win_end, "window[0] must be less than window[1]."

    # Negative windows: mirror to positive and transpose at the end
    transpose_result = False
    if win_end <= 0:
        win_start, win_end = -win_end, -win_start
        transpose_result = True

    if spike_clusters is None:
        spike_clusters = np.zeros(len(spike_times), dtype=np.int32)
    else:
        spike_clusters = np.asarray(spike_clusters, dtype=np.int32)

    assert len(spike_times) == len(spike_clusters), \
        "spike_times and spike_clusters must have the same length."

    if not np.all(np.diff(spike_times) >= 0):
        sort_inds = np.argsort(spike_times)
        spike_times = spike_times[sort_inds]
        spike_clusters = spike_clusters[sort_inds]

    # --- 2. Cluster filtering and mapping (CPU) ---
    if uids is None:
        uids = np.unique(spike_clusters)
        needs_reordering = False
    else:
        uids = np.asarray(uids, dtype=np.int32)
        needs_reordering = True

    membership_mask = np.isin(spike_clusters, uids)
    spike_times = spike_times[membership_mask]
    spike_clusters = spike_clusters[membership_mask]

    n_clusters = len(uids)
    unsort_indices = None

    if needs_reordering:
        sort_indices = np.argsort(uids)
        sorted_uids = uids[sort_indices]
        unsort_indices = np.argsort(sort_indices)
    else:
        sorted_uids = uids

    spike_inds = np.searchsorted(sorted_uids, spike_clusters).astype(np.int32)

    # --- 3. Select backend ---
    if device == "cpu":
        xp = np
    else:
        import cupy as cp
        xp = cp
        spike_times = cp.asarray(spike_times)
        spike_inds = cp.asarray(spike_inds)

    n_spikes = len(spike_times)
    counts = xp.zeros((n_clusters, n_clusters), dtype=xp.int32)
    n_sq = n_clusters * n_clusters

    # --- 4. Shift-based accumulation (forward shifts only) ---
    mask = xp.ones(n_spikes, dtype=bool)
    shift = 1

    while True:
        mask[n_spikes - shift:] = False
        active = xp.where(mask[:n_spikes - shift])[0]
        if len(active) == 0:
            break

        dts = spike_times[active + shift] - spike_times[active]
        in_window = (dts > win_start) & (dts < win_end)

        if xp.any(in_window):
            w = active[in_window]
            ravel = (spike_inds[w].astype(xp.int64) * n_clusters
                     + spike_inds[w + shift].astype(xp.int64))
            flat = xp.bincount(ravel, minlength=n_sq)
            counts += flat[:n_sq].reshape(n_clusters, n_clusters).astype(xp.int32)

        # Deactivate references whose dt already exceeds the window
        mask[active] = dts < win_end
        shift += 1

    # --- 5. Transfer back and reorder ---
    result = counts
    if device != "cpu":
        result = result.get()

    if needs_reordering:
        result = result[unsort_indices][:, unsort_indices]

    if transpose_result:
        result = result.T.copy()

    return result
