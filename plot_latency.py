"""
AetherFlow Latency Distribution & CDF Plotter
Generates publishable benchmark graph for GitHub repository.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from features import compute_window_features
from market_generator import generate_batch

WINDOW_SIZE = 1_000
ITERATIONS = 50_000
ALPHA = 0.05


def main():
    print(f"Collecting {ITERATIONS:,} latency samples across {WINDOW_SIZE}-tick sliding windows...")
    ticks = generate_batch(WINDOW_SIZE)
    bid_p = np.ascontiguousarray(ticks['bid_price'])
    ask_p = np.ascontiguousarray(ticks['ask_price'])
    bid_s = np.ascontiguousarray(ticks['bid_size'])
    ask_s = np.ascontiguousarray(ticks['ask_size'])

    # Warmup
    _ = compute_window_features(bid_p, ask_p, bid_s, ask_s, ALPHA)

    # Collect high-res nanosecond timings
    latencies_ns = np.empty(ITERATIONS, dtype=np.int64)
    for i in range(ITERATIONS):
        t0 = time.perf_counter_ns()
        _ = compute_window_features(bid_p, ask_p, bid_s, ask_s, ALPHA)
        latencies_ns[i] = time.perf_counter_ns() - t0

    latencies_us = latencies_ns / 1_000.0
    sorted_lat = np.sort(latencies_us)
    cdf = np.arange(1, ITERATIONS + 1) / ITERATIONS

    # Generate Chart
    plt.figure(figsize=(9, 5))
    plt.plot(sorted_lat, cdf, color="#00E5FF", linewidth=2, label="AetherFlow Fused Kernel")
    plt.axvline(np.percentile(latencies_us, 50), color="#76FF03", linestyle="--", label=f"p50: {np.percentile(latencies_us, 50):.2f} µs")
    plt.axvline(np.percentile(latencies_us, 99), color="#FF3D00", linestyle="--", label=f"p99: {np.percentile(latencies_us, 99):.2f} µs")

    plt.xscale("log")
    plt.title("AetherFlow Microstructure Signal Latency Distribution (1,000-Tick Window)", fontsize=12, fontweight="bold")
    plt.xlabel("Latency (Microseconds, Log Scale)", fontsize=10)
    plt.ylabel("Cumulative Probability (CDF)", fontsize=10)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig("benchmark_latency.png", dpi=300)
    print("Benchmark plot saved to 'benchmark_latency.png'.")


if __name__ == "__main__":
    main()