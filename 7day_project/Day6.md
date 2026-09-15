## Day 6: Benchmarking & Profiling (4 Hours)

Goal: Generate the concrete numbers you will put on your resume.

Tasks:

Build a baseline script that does the same work using pure Python loops
and multiprocessing.Queue.

Benchmark your shared-memory + Numba engine against the baseline.

Record:

Throughput: Ticks processed per second (aim for >100,000 ticks/sec).

Latency: Percentile processing latency (p50, p99 in microseconds).

Use matplotlib to plot a benchmark graph for your GitHub README.md.



### Manual: benchmark_comparison.py, plot_latency.py



BENCHMARK COMPARISON
┌──────────────────────────────────────────────────────────────┐
│  Baseline Architecture:                                      │
│  multiprocessing.Queue + pickle + pure Python loops          │
│  Throughput: ~35,000 ticks/sec | Latency: 150-400 µs (GC jitter)
└──────────────────────────────────────────────────────────────┘
                              VS
┌──────────────────────────────────────────────────────────────┐
│  AetherFlow Architecture:                                    │
│  Zero-Copy POSIX Shared Memory + Fused Numba Kernels         │
│  Throughput: 3,500,000+ ticks/sec | Latency: 4.5 µs (No GC)   │
└──────────────────────────────────────────────────────────────┘