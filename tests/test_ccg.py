"""Tests for spike_utils.ccg — calc_ccgs, calc_single_bin_ccgs, calc_local_firing_rate."""

import numpy as np
import pytest

from spike_utils import calc_ccgs, calc_local_firing_rate, calc_single_bin_ccgs


def _has_cupy():
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spike_data():
    """Synthetic spike data: 10 clusters, 50k spikes over 60s."""
    rng = np.random.default_rng(42)
    n_spikes = 50_000
    n_clusters = 10
    duration = 60.0
    spike_times = np.sort(rng.uniform(0, duration, size=n_spikes))
    spike_clusters = rng.integers(0, n_clusters, size=n_spikes).astype(np.int32)
    uids = np.arange(n_clusters, dtype=np.int32)
    return spike_times, spike_clusters, uids


# ---------------------------------------------------------------------------
# calc_local_firing_rate
# ---------------------------------------------------------------------------

class TestCalcLocalFiringRate:
    def test_empty(self):
        assert calc_local_firing_rate(np.array([])) == 0.0

    def test_poisson_rate(self):
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 100, size=2000))
        fr = calc_local_firing_rate(times, 1.0)
        assert 15 < fr < 25, f"Expected ~20 Hz, got {fr}"

    def test_single_spike(self):
        assert calc_local_firing_rate(np.array([1.0])) == 0.0


# ---------------------------------------------------------------------------
# calc_single_bin_ccgs
# ---------------------------------------------------------------------------

class TestCalcSingleBinCcgs:
    def test_matches_calc_ccgs(self, spike_data):
        """Single-bin CCG must match calc_ccgs with the same window."""
        spike_times, spike_clusters, uids = spike_data
        window = (0.5e-3, 2.5e-3)

        ref = calc_ccgs(
            spike_times, [window[0], window[1]], spike_clusters, uids=uids
        ).squeeze(axis=-1)
        result = calc_single_bin_ccgs(
            spike_times, window, spike_clusters, uids=uids
        )

        np.testing.assert_array_equal(result, ref)

    def test_transpose_symmetry(self, spike_data):
        """Right-bin result.T should equal left-bin result."""
        spike_times, spike_clusters, uids = spike_data
        right = (0.5e-3, 2.5e-3)
        left = (-2.5e-3, -0.5e-3)

        right_result = calc_single_bin_ccgs(
            spike_times, right, spike_clusters, uids=uids
        )
        left_result = calc_single_bin_ccgs(
            spike_times, left, spike_clusters, uids=uids
        )

        np.testing.assert_array_equal(right_result.T, left_result)

    def test_no_spikes_in_window(self, spike_data):
        """Window far from any ISI should return all zeros."""
        spike_times, spike_clusters, uids = spike_data
        result = calc_single_bin_ccgs(
            spike_times, (100.0, 200.0), spike_clusters, uids=uids
        )
        assert result.sum() == 0

    def test_single_cluster(self):
        """Should work with a single cluster (autocorrelogram-like)."""
        times = np.array([0.0, 0.001, 0.002, 0.005, 0.010])
        result = calc_single_bin_ccgs(times, (0.5e-3, 2.5e-3))
        assert result.shape == (1, 1)
        assert result[0, 0] > 0

    def test_unsorted_input(self, spike_data):
        """Should handle unsorted spike times correctly."""
        spike_times, spike_clusters, uids = spike_data
        window = (0.5e-3, 2.5e-3)

        ref = calc_single_bin_ccgs(spike_times, window, spike_clusters, uids=uids)

        rng = np.random.default_rng(99)
        perm = rng.permutation(len(spike_times))
        result = calc_single_bin_ccgs(
            spike_times[perm], window, spike_clusters[perm], uids=uids
        )

        np.testing.assert_array_equal(result, ref)

    def test_uid_subset(self, spike_data):
        """Should correctly filter to a subset of uids."""
        spike_times, spike_clusters, uids = spike_data
        window = (0.5e-3, 2.5e-3)
        subset = uids[::2]

        ref = calc_ccgs(
            spike_times, [window[0], window[1]], spike_clusters, uids=subset
        ).squeeze(axis=-1)
        result = calc_single_bin_ccgs(
            spike_times, window, spike_clusters, uids=subset
        )

        np.testing.assert_array_equal(result, ref)

    @pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
    def test_gpu_matches_cpu(self, spike_data):
        """GPU result must match CPU result."""
        spike_times, spike_clusters, uids = spike_data
        window = (0.5e-3, 2.5e-3)

        cpu = calc_single_bin_ccgs(spike_times, window, spike_clusters, uids=uids, device="cpu")
        gpu = calc_single_bin_ccgs(spike_times, window, spike_clusters, uids=uids, device="cuda")
        np.testing.assert_array_equal(gpu, cpu)
