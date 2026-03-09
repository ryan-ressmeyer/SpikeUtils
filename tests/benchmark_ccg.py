#%% Benchmark: calc_ccgs CPU (NumPy) vs GPU (CuPy)
import time

import numpy as np

from spike_utils import calc_ccgs

#%% Generate synthetic spike data
rng = np.random.default_rng(42)

n_clusters = 20
n_spikes = 500_000
duration = 600.0  # 10 minutes of recording

spike_times = np.sort(rng.uniform(0, duration, size=n_spikes))
spike_clusters = rng.integers(0, n_clusters, size=n_spikes)

# +/- 50 ms window, 1 ms bins
bin_edges = np.linspace(-0.05, 0.05, 101)

print(f"Spikes: {n_spikes:,}")
print(f"Clusters: {n_clusters}")
print(f"Bins: {len(bin_edges) - 1}")

try:
    import cupy as cp
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    has_gpu = True
except ImportError:
    print("CuPy not available, skipping GPU benchmark")
    has_gpu = False
print()

#%% Run CPU version
t0 = time.perf_counter()
ccgs_cpu = calc_ccgs(spike_times, bin_edges, spike_clusters, progress=True, device="cpu")
t_cpu = time.perf_counter() - t0
print(f"CPU (NumPy):  {t_cpu:.2f} s\n")

#%% Run GPU version
if has_gpu:
    # Warm up
    _ = calc_ccgs(spike_times[:1000], bin_edges, device="cuda")

    t0 = time.perf_counter()
    ccgs_gpu = calc_ccgs(spike_times, bin_edges, spike_clusters, progress=True, device="cuda")
    t_gpu = time.perf_counter() - t0
    print(f"GPU (CuPy):   {t_gpu:.2f} s\n")

    # Verify correctness
    match = np.array_equal(ccgs_cpu, ccgs_gpu)
    if not match:
        max_diff = np.max(np.abs(ccgs_cpu.astype(np.int64) - ccgs_gpu.astype(np.int64)))
        print(f"MISMATCH — max absolute difference: {max_diff}")
    else:
        print("Results match exactly.")

    print(f"\nSpeedup: {t_cpu / t_gpu:.1f}x")
