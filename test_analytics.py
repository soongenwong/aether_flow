"""Verification and performance harness for :mod:`analytics`."""

from __future__ import annotations

from time import perf_counter_ns, time_ns

import numpy as np
import pandas as pd

from analytics import AnalyticsEngine
from market_generator import generate_batch
from ring_buffer import SharedMemoryRingBuffer


SYMBOLS = ("BTC", "ETH", "SOL")
BUFFER_CAPACITY = 65_536
INITIAL_TICKS = 10_000
BENCHMARK_TICKS = 50_000
TICK_INTERVAL_NS = 10_000_000  # 10 ms; 10,000 ticks span approximately 100 s.
BENCHMARK_TARGET_MS = 15.0
TIMEFRAMES = ("100ms", "1s", "5s")
BUFFER_NAMES = {
    symbol: f"aetherflow_analytics_{symbol.lower()}" for symbol in SYMBOLS
}


def _timestamp_batch(ticks: np.ndarray, start_ns: int) -> np.ndarray:
    """Assign a realistic, aligned 10 ms timestamp grid in place."""

    ticks["timestamp_ns"] = np.int64(start_ns) + (
        np.arange(len(ticks), dtype=np.int64) * np.int64(TICK_INTERVAL_NS)
    )
    return ticks


def _expected_resampled_volume(
    ticks: np.ndarray,
    timeframe: str,
) -> tuple[pd.Series, pd.Series]:
    timestamps = pd.to_datetime(ticks["timestamp_ns"], unit="ns")
    total_size = pd.Series(
        ticks["bid_size"] + ticks["ask_size"],
        index=timestamps,
        name="total_size",
    )
    tick_count = pd.Series(1, index=timestamps, name="tick_count")
    return (
        total_size.resample(timeframe).sum().dropna(),
        tick_count.resample(timeframe).sum().dropna(),
    )


def _assert_ohlcv_invariants(
    bars: pd.DataFrame,
    ticks: np.ndarray,
    timeframe: str,
    symbol: str,
) -> None:
    assert not bars.empty, f"{symbol} produced no {timeframe} bars"
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["open"]).all()
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["open"]).all()
    assert (bars["low"] <= bars["close"]).all()

    expected_volume, expected_count = _expected_resampled_volume(ticks, timeframe)
    assert bars.index.equals(expected_volume.index), (
        f"{symbol} {timeframe} bar index differs from expected resampling"
    )
    np.testing.assert_allclose(
        bars["volume"].to_numpy(),
        expected_volume.to_numpy(),
        rtol=1.0e-12,
        atol=1.0e-9,
    )
    np.testing.assert_array_equal(
        bars["tick_count"].to_numpy(),
        expected_count.to_numpy(),
    )


def _run_ohlcv_checks(
    engine: AnalyticsEngine,
    batches: dict[str, np.ndarray],
) -> None:
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            bars = engine.get_ohlcv(symbol, timeframe=timeframe)
            _assert_ohlcv_invariants(
                bars,
                batches[symbol],
                timeframe,
                symbol,
            )


def _run_correlation_check(engine: AnalyticsEngine) -> None:
    correlation = engine.compute_rolling_correlation(
        list(SYMBOLS),
        timeframe="1s",
        rolling_window=20,
        last_n_ticks=INITIAL_TICKS,
    )
    matrix = correlation.loc[list(SYMBOLS), list(SYMBOLS)].to_numpy()

    assert np.all(np.isfinite(matrix)), "correlation matrix contains non-finite values"
    assert np.all(np.diag(matrix) == 1.0), "correlation diagonal is not exactly 1.0"
    assert np.array_equal(matrix, matrix.T), "correlation matrix is not symmetric"


def _run_benchmark(
    engine: AnalyticsEngine,
    benchmark_ticks: np.ndarray,
) -> None:
    # Warm Pandas' resampling path before timing the requested 50,000-tick call.
    engine.get_ohlcv("BTC", timeframe="1s", last_n_ticks=BENCHMARK_TICKS)

    started_ns = perf_counter_ns()
    bars = engine.get_ohlcv(
        "BTC",
        timeframe="1s",
        last_n_ticks=BENCHMARK_TICKS,
    )
    elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000.0
    assert not bars.empty
    assert len(benchmark_ticks) == BENCHMARK_TICKS

    status = "PASS" if elapsed_ms < BENCHMARK_TARGET_MS else "MISS"
    print(f"get_ohlcv({BENCHMARK_TICKS:,} ticks): {elapsed_ms:.3f} ms [{status}]")


def main() -> None:
    buffers: dict[str, SharedMemoryRingBuffer] = {}
    try:
        start_ns = (time_ns() // 1_000_000_000) * 1_000_000_000
        batches: dict[str, np.ndarray] = {}
        start_prices = {"BTC": 100.0, "ETH": 3_000.0, "SOL": 150.0}

        for offset, symbol in enumerate(SYMBOLS):
            buffer = SharedMemoryRingBuffer(
                name=BUFFER_NAMES[symbol],
                capacity=BUFFER_CAPACITY,
                create=True,
            )
            buffers[symbol] = buffer

            ticks = generate_batch(
                INITIAL_TICKS,
                start_price=start_prices[symbol],
                rng=np.random.default_rng(10_000 + offset),
            )
            batches[symbol] = _timestamp_batch(ticks, start_ns)
            buffer.write_batch(batches[symbol])

        engine = AnalyticsEngine(buffers)
        _run_ohlcv_checks(engine, batches)
        _run_correlation_check(engine)

        # Extend BTC after correctness checks so the benchmark exercises the
        # requested 50,000-tick read while the correlation test uses aligned
        # 10,000-tick histories for all three assets.
        btc_last_timestamp = int(batches["BTC"]["timestamp_ns"][-1])
        extra_ticks = generate_batch(
            BENCHMARK_TICKS - INITIAL_TICKS,
            start_price=start_prices["BTC"],
            rng=np.random.default_rng(20_026),
        )
        _timestamp_batch(extra_ticks, btc_last_timestamp + TICK_INTERVAL_NS)
        buffers["BTC"].write_batch(extra_ticks)

        benchmark_ticks = np.concatenate((batches["BTC"], extra_ticks))
        _run_benchmark(engine, benchmark_ticks)
        print("OHLCV correctness: PASS")
        print("correlation correctness: PASS")
    finally:
        for buffer in buffers.values():
            try:
                buffer.close()
            finally:
                buffer.unlink()


if __name__ == "__main__":
    main()
