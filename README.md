# AetherFlow

> **High-Throughput Streaming Market Data & Microstructure Feature Engine**  
> Zero-copy POSIX shared memory, pre-allocated power-of-2 ring buffers, and sub-5 µs compiled feature kernels.

---

## Overview

AetherFlow bypasses Python's runtime bottlenecks (GIL, garbage collector churn, and IPC serialization overhead) to process live order-book feeds and generate microstructure trading signals in microseconds.

- **Zero-Copy IPC (`multiprocessing.shared_memory`):** Maps identical physical RAM across isolated processes to eliminate `pickle` serialization.
- **Cache-Optimized Ring Buffer:** $2^{20}$-slot circular buffer using single-cycle bitwise modulo masking (`seq & (N - 1)`) with a 64-byte padded header to prevent false sharing.
- **Fused Numba JIT Engine:** Release the GIL (`@njit(nogil=True)`) to compute Order Flow Imbalance (OFI), Realized Volatility, and EWMA in a single memory pass with zero heap allocations.
- **Analytical API:** Slices live memory into multi-timeframe OHLCV bars and rolling correlation matrices via Pandas.

---

## Verified Benchmarks

| Component | Target Metric | AetherFlow Result | vs. Standard Python |
| :--- | :--- | :--- | :--- |
| **Synthetic Feed** | Batch Generation Rate | **7,689,335 ticks/sec** | Instant vectorized feed |
| **Shared Memory IPC** | Write Bandwidth | **47,512,522 ticks/sec** | **~1,050x** vs `mp.Queue` |
| **Shared Memory IPC** | End-to-End Throughput | **3,535,820 ticks/sec** | **~101x** vs `mp.Queue` |
| **Feature Kernel** | Mean Latency (1k-tick window) | **4.569 µs** (4.57 ns/tick) | **~209k windows/sec** |
| **Latency Stability** | Tail Jitter (p99 vs p50) | **4.750 µs vs 4.500 µs** | **< 250 ns variance** (No GC) |

---

## 7-Day Roadmap

- [x] **Day 1: Binary Protocol & Feed** — 40-byte aligned struct; 7.68M tick/s Poisson-GBM generator.
- [x] **Day 2: Shared Memory Ring Buffer** — Zero-copy buffer; 47.5M write / 3.53M e2e IPC ticks/sec.
- [x] **Day 3: Numba Feature Compute** — Fused OFI, Volatility, and EWMA kernels running at 4.56 µs.
- [ ] **Day 4: Analytical Layer** — Resampling ring buffer into multi-timeframe OHLCV & rolling correlations.
- [ ] **Day 5: Multi-Process Pipeline** — Producer, compute workers, and query engine with graceful POSIX lifecycle.
- [ ] **Day 6: Benchmarking & Profiling** — Profiling scripts proving zero GC churn; queue vs. SHM comparison.
- [ ] **Day 7: Documentation & Release** — Matplotlib benchmark visualizations and CV packaging.

---

## Quickstart

```bash
# 1. Verify 40-byte memory layout and pointer views
python verify_layout.py

# 2. Test synthetic tick generation
python market_generator.py

# 3. Benchmark multi-process shared memory IPC
python test_ring_buffer.py

# 4. Benchmark sub-5 µs feature kernel latency
python test_features.py