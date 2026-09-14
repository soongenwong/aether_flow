"""Correctness checks and low-latency benchmark for the feature kernels."""

from __future__ import annotations

from time import perf_counter_ns

import numpy as np

from features import (
    compute_ofi_tick,
    compute_rolling_ewma,
    compute_rolling_realized_vol,
    compute_window_features,
)
from market_generator import generate_batch


WINDOW_SIZE = 1_000
BENCHMARK_ITERATIONS = 50_000
EWMA_ALPHA = 0.20
TARGET_MEAN_LATENCY_US = 5.0


def warmup_numba() -> None:
    """Compile every Numba entry point before collecting measurements."""

    prices = np.linspace(99.99, 100.01, 8, dtype=np.float64)
    sizes = np.full(prices.shape, 10.0, dtype=np.float64)

    compute_ofi_tick(
        prices[1],
        prices[0],
        sizes[1],
        sizes[0],
        prices[1] + 0.01,
        prices[0] + 0.01,
        sizes[1],
        sizes[0],
    )
    compute_rolling_realized_vol(prices)
    compute_rolling_ewma(prices, EWMA_ALPHA)
    compute_window_features(prices, prices + 0.01, sizes, sizes, EWMA_ALPHA)


def test_ofi_logic() -> None:
    """Verify OFI for three same-price ticks by paper calculation."""

    bid_prices = np.array([100.00, 100.00, 100.00], dtype=np.float64)
    ask_prices = np.array([100.01, 100.01, 100.01], dtype=np.float64)
    bid_sizes = np.array([10.0, 15.0, 18.0], dtype=np.float64)
    ask_sizes = np.array([20.0, 24.0, 30.0], dtype=np.float64)

    manual_ofi = np.array(
        [
            (15.0 - 10.0) - (24.0 - 20.0),
            (18.0 - 15.0) - (30.0 - 24.0),
        ],
        dtype=np.float64,
    )
    calculated_ofi = np.array(
        [
            compute_ofi_tick(
                bid_prices[i],
                bid_prices[i - 1],
                bid_sizes[i],
                bid_sizes[i - 1],
                ask_prices[i],
                ask_prices[i - 1],
                ask_sizes[i],
                ask_sizes[i - 1],
            )
            for i in range(1, len(bid_prices))
        ],
        dtype=np.float64,
    )

    assert np.array_equal(calculated_ofi, manual_ofi), (
        f"OFI mismatch: {calculated_ofi} != {manual_ofi}"
    )
    total_ofi, _, _ = compute_window_features(
        bid_prices,
        ask_prices,
        bid_sizes,
        ask_sizes,
        EWMA_ALPHA,
    )
    assert total_ofi == float(manual_ofi.sum())


def test_constant_price_volatility() -> None:
    """A constant positive price series has zero log-return volatility."""

    prices = np.full(WINDOW_SIZE, 100.0, dtype=np.float64)
    volatility = compute_rolling_realized_vol(prices)
    assert volatility == 0.0, f"constant-price volatility was {volatility}"


def benchmark_window_features() -> None:
    """Benchmark the fused feature calculation and print latency statistics."""

    ticks = generate_batch(
        WINDOW_SIZE,
        start_price=100.0,
        rng=np.random.default_rng(2026),
    )
    bid_prices = np.ascontiguousarray(ticks["bid_price"], dtype=np.float64)
    ask_prices = np.ascontiguousarray(ticks["ask_price"], dtype=np.float64)
    bid_sizes = np.ascontiguousarray(ticks["bid_size"], dtype=np.float64)
    ask_sizes = np.ascontiguousarray(ticks["ask_size"], dtype=np.float64)

    # The warmup call above excludes LLVM compilation from this measurement;
    # keep the result live so the benchmark cannot be optimized away.
    sink = compute_window_features(
        bid_prices,
        ask_prices,
        bid_sizes,
        ask_sizes,
        EWMA_ALPHA,
    )
    latencies_ns = np.empty(BENCHMARK_ITERATIONS, dtype=np.int64)

    started_ns = perf_counter_ns()
    for iteration in range(BENCHMARK_ITERATIONS):
        call_started_ns = perf_counter_ns()
        sink = compute_window_features(
            bid_prices,
            ask_prices,
            bid_sizes,
            ask_sizes,
            EWMA_ALPHA,
        )
        latencies_ns[iteration] = perf_counter_ns() - call_started_ns
    elapsed_seconds = (perf_counter_ns() - started_ns) / 1_000_000_000.0

    assert np.all(np.isfinite(np.asarray(sink, dtype=np.float64)))
    latencies_us = latencies_ns.astype(np.float64) / 1_000.0
    mean_latency_us = float(np.mean(latencies_us))
    p50_us, p90_us, p99_us = np.percentile(latencies_us, [50.0, 90.0, 99.0])
    calculations_per_second = BENCHMARK_ITERATIONS / elapsed_seconds

    target_status = "PASS" if mean_latency_us < TARGET_MEAN_LATENCY_US else "MISS"
    print(f"window size:           {WINDOW_SIZE:,} ticks")
    print(f"iterations:            {BENCHMARK_ITERATIONS:,}")
    print(f"mean latency:          {mean_latency_us:.3f} us [{target_status}]")
    print(f"p50 latency:           {p50_us:.3f} us")
    print(f"p90 latency:           {p90_us:.3f} us")
    print(f"p99 latency:           {p99_us:.3f} us")
    print(f"calculations/second:    {calculations_per_second:,.0f}")


def main() -> None:
    warmup_numba()
    test_ofi_logic()
    test_constant_price_volatility()
    print("correctness:            PASS")
    benchmark_window_features()


if __name__ == "__main__":
    main()
