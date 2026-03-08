from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import tqdm


def calc_ccgs(
    spike_times: np.ndarray,
    bin_edges: np.ndarray,
    spike_clusters: Optional[np.ndarray] = None,
    uids: Optional[np.ndarray] = None,
    progress: bool = False,
) -> np.ndarray:
    """
    Compute all pairwise cross-correlograms among a set of spike clusters.

    This function uses a highly efficient, vectorized algorithm that iterates
    by "shift" in the spike array rather than by individual spike pairs. It
    skips counting spike pairs with a time lag of zero, meaning the central
    bin of an autocorrelogram will not equal the total spike count.

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

    Returns
    -------
    correlograms : np.ndarray
        A 3D array of shape `(n_clusters, n_clusters, n_bins)` containing
        integer counts. `correlograms[i, j, k]` is the number of spikes from
        cluster `uids[j]` that occurred in time bin `k` relative to a spike
        from cluster `uids[i]`.
    """
    # --- 1. Input Validation and Preparation ---
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

    # The algorithm requires sorted spike times
    if not np.all(np.diff(spike_times) >= 0):
        print("Spike times are not sorted, sorting...")
        sort_inds = np.argsort(spike_times)
        spike_times = spike_times[sort_inds]
        spike_clusters = spike_clusters[sort_inds]

    # --- 2. Cluster ID Filtering and Mapping ---
    if uids is None:
        uids = np.unique(spike_clusters)
        needs_reordering = False
    else:
        uids = np.asarray(uids, dtype=np.int32)
        needs_reordering = True

    # Filter spikes to only include those in the `uids` list.
    membership_mask = np.isin(spike_clusters, uids)
    spike_times = spike_times[membership_mask]
    spike_clusters = spike_clusters[membership_mask]

    n_spikes = len(spike_times)
    n_clusters = len(uids)

    # Map raw cluster IDs to 0-based indices using the fast, sorted method
    if needs_reordering:
        sort_indices = np.argsort(uids)
        sorted_uids = uids[sort_indices]
        unsort_indices = np.argsort(sort_indices)
    else:
        sorted_uids = uids  # Already sorted and unique

    spike_inds = np.searchsorted(sorted_uids, spike_clusters)

    # --- 3. Binning Setup ---
    n_bins = len(bin_edges) - 1
    ccgs = np.zeros((n_clusters, n_clusters, n_bins), dtype=np.int32)
    min_bin, max_bin = bin_edges[0], bin_edges[-1]

    # Optimization: Use a faster calculation for linearly spaced bins
    bin_diffs = np.diff(bin_edges)
    if np.allclose(bin_diffs, bin_diffs[0]):
        bin_width = bin_diffs[0]
        def digitize(x):
            bins = np.asarray((x - min_bin) / bin_width, dtype=np.int32)
            return np.clip(bins, 0, n_bins - 1)
    else:
        def digitize(x):
            return np.digitize(x, bin_edges) - 1

    # --- 4. Main Correlogram Loop ---
    shift = 1
    pos_mask = np.ones(n_spikes, dtype=bool)
    neg_mask = np.ones(n_spikes, dtype=bool)
    pbar = tqdm(total=1.0, desc="Calculating CCGs", disable=not progress)

    while True:
        # --- Positive Lags (Ref: i, Target: i + shift) ---
        pos_mask[-shift:] = False
        active_pos_indices = np.where(pos_mask[:-shift])[0]
        has_pos = len(active_pos_indices) > 0

        if has_pos:
            pos_dts = spike_times[active_pos_indices + shift] - spike_times[active_pos_indices]
            valid_pos = (min_bin < pos_dts) & (pos_dts < max_bin)
            if np.any(valid_pos):
                valid_indices = active_pos_indices[valid_pos]
                pos_i, pos_j = spike_inds[valid_indices], spike_inds[valid_indices + shift]
                pos_bins = digitize(pos_dts[valid_pos])
                ravel_inds = np.ravel_multi_index((pos_i, pos_j, pos_bins), ccgs.shape)
                counts = np.bincount(ravel_inds, minlength=ccgs.size)
                ccgs += counts.reshape(ccgs.shape)
            pos_mask[active_pos_indices] = pos_dts < max_bin

        # --- Negative Lags (Ref: i + shift, Target: i) ---
        neg_mask[:shift] = False
        active_neg_indices = np.where(neg_mask[shift:])[0]
        has_neg = len(active_neg_indices) > 0

        if has_neg:
            neg_dts = spike_times[active_neg_indices] - spike_times[active_neg_indices + shift]
            valid_neg = (min_bin < neg_dts) & (neg_dts < max_bin)
            if np.any(valid_neg):
                valid_indices = active_neg_indices[valid_neg]
                neg_i, neg_j = spike_inds[valid_indices + shift], spike_inds[valid_indices]
                neg_bins = digitize(neg_dts[valid_neg])
                ravel_inds = np.ravel_multi_index((neg_i, neg_j, neg_bins), ccgs.shape)
                counts = np.bincount(ravel_inds, minlength=ccgs.size)
                ccgs += counts.reshape(ccgs.shape)
            neg_mask[active_neg_indices + shift] = neg_dts > min_bin

        # --- Loop Control and Progress Update ---
        if not has_pos and not has_neg:
            pbar.n = 1.0
            pbar.set_description("Calculating CCGs: Done")
            break

        if progress:
            progress_val = 1.0 - (np.sum(pos_mask) + np.sum(neg_mask)) / (2 * n_spikes)
            pbar.n = np.round(progress_val, 3)
            pbar.set_description(f"Calculating CCGs: Shift {shift}")

        shift += 1

    pbar.close()

    # Reorder axes to match original `uids` order if necessary
    if needs_reordering:
        ccgs = ccgs[unsort_indices][:, unsort_indices]

    return ccgs
