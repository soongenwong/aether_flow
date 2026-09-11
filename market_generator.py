"""High-throughput synthetic market tick generation.

The batch path uses NumPy vectorization throughout.  The streaming path keeps
only the current market state in Python and uses a monotonic-clock schedule so
that callers can consume an effectively unbounded stream at a requested rate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, exp, sqrt
from time import perf_counter_ns, sleep, time_ns
from typing import Iterator, Literal

import numpy as np


TICK_DTYPE = np.dtype(
    [
        ("timestamp_ns", np.int64),
        ("bid_price", np.float64),
        ("ask_price", np.float64),
        ("bid_size", np.float64),
        ("ask_size", np.float64),
    ],
    align=True,
)

Tick = tuple[int, float, float, float, float]
VolumeDistribution = Literal["gamma", "lognormal"]


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Parameters for the synthetic market process.

    Rates are expressed per second.  ``sample_interval_seconds`` is used by
    :meth:`MarketGenerator.generate_batch`; the streaming method derives its
    interval from ``target_rate``.
    """

    tick_size: float = 0.01
    sample_interval_seconds: float = 1.0e-5

    # GBM log-return parameters.
    drift_per_second: float = 0.0
    volatility_per_sqrt_second: float = 0.20
    jump_intensity_per_second: float = 0.25
    jump_mean: float = 0.0
    jump_volatility: float = 0.01

    # Spread is an OU-like mean-reverting process in tick units.
    baseline_spread_ticks: float = 1.5
    spread_reversion_per_second: float = 8.0
    spread_volatility_ticks_per_sqrt_second: float = 1.5
    minimum_spread_ticks: float = 1.0

    # Queue-size distribution parameters.
    volume_distribution: VolumeDistribution = "lognormal"
    volume_shape: float = 2.0
    volume_scale: float = 50.0
    volume_log_mean: float = 4.0
    volume_log_sigma: float = 0.70

    def __post_init__(self) -> None:
        if self.tick_size <= 0.0 or not np.isfinite(self.tick_size):
            raise ValueError("tick_size must be finite and greater than zero")
        if self.sample_interval_seconds <= 0.0 or not np.isfinite(
            self.sample_interval_seconds
        ):
            raise ValueError(
                "sample_interval_seconds must be finite and greater than zero"
            )
        if (
            not np.isfinite(self.volatility_per_sqrt_second)
            or self.volatility_per_sqrt_second < 0.0
        ):
            raise ValueError(
                "volatility_per_sqrt_second must be finite and non-negative"
            )
        if (
            not np.isfinite(self.jump_intensity_per_second)
            or self.jump_intensity_per_second < 0.0
        ):
            raise ValueError(
                "jump_intensity_per_second must be finite and non-negative"
            )
        if not np.isfinite(self.jump_mean):
            raise ValueError("jump_mean must be finite")
        if not np.isfinite(self.jump_volatility) or self.jump_volatility < 0.0:
            raise ValueError("jump_volatility must be finite and non-negative")
        if (
            not np.isfinite(self.baseline_spread_ticks)
            or self.baseline_spread_ticks <= 0.0
        ):
            raise ValueError("baseline_spread_ticks must be greater than zero")
        if (
            not np.isfinite(self.spread_reversion_per_second)
            or self.spread_reversion_per_second < 0.0
        ):
            raise ValueError("spread_reversion_per_second must be finite and non-negative")
        if (
            not np.isfinite(self.spread_volatility_ticks_per_sqrt_second)
            or self.spread_volatility_ticks_per_sqrt_second < 0.0
        ):
            raise ValueError(
                "spread_volatility_ticks_per_sqrt_second must be finite and non-negative"
            )
        if (
            not np.isfinite(self.minimum_spread_ticks)
            or self.minimum_spread_ticks <= 0.0
        ):
            raise ValueError("minimum_spread_ticks must be greater than zero")
        if self.volume_distribution not in ("gamma", "lognormal"):
            raise ValueError("volume_distribution must be 'gamma' or 'lognormal'")
        if not np.isfinite(self.volume_shape) or not np.isfinite(self.volume_scale):
            raise ValueError("gamma shape and scale must be finite")
        if self.volume_shape <= 0.0 or self.volume_scale <= 0.0:
            raise ValueError("gamma shape and scale must be greater than zero")
        if not np.isfinite(self.volume_log_mean):
            raise ValueError("volume_log_mean must be finite")
        if not np.isfinite(self.volume_log_sigma) or self.volume_log_sigma < 0.0:
            raise ValueError("volume_log_sigma must be finite and non-negative")


def _resolve_config(
    config: MarketConfig | None,
    tick_size: float | None,
) -> MarketConfig:
    resolved = config if config is not None else MarketConfig()
    return replace(resolved, tick_size=tick_size) if tick_size is not None else resolved


def _validate_start_price(start_price: float) -> float:
    value = float(start_price)
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("start_price must be finite and greater than zero")
    return value


def _mean_reverting_path(
    noise: np.ndarray,
    baseline: float,
    initial: float,
    reversion: float,
    dt: float,
) -> np.ndarray:
    """Return an OU-like path using vectorized recurrence blocks.

    A direct Python recurrence would dominate the batch runtime.  Each block
    evaluates the AR(1) closed form with NumPy operations and carries only its
    final state into the next block.
    """

    size = noise.size
    result = np.empty(size, dtype=np.float64)
    if size == 0:
        return result

    rho = float(np.exp(-reversion * dt))
    initial_deviation = initial - baseline
    if rho == 0.0:
        result[:] = baseline + noise
        return result
    if rho == 1.0:
        result[:] = baseline + initial_deviation + np.cumsum(noise)
        return result

    # A short block keeps rho**-index numerically stable for every valid
    # configuration, including unusually large sampling intervals.
    block_size = 1024
    powers = np.power(rho, np.arange(1, block_size + 1, dtype=np.float64))
    inverse_powers = 1.0 / powers
    previous = initial_deviation

    for start in range(0, size, block_size):
        stop = min(start + block_size, size)
        length = stop - start
        block_powers = powers[:length]
        block = noise[start:stop]
        previous_path = block_powers * previous
        previous_path += block_powers * np.cumsum(block * inverse_powers[:length])
        result[start:stop] = baseline + previous_path
        previous = float(previous_path[-1])

    return result


def _generate_prices_and_spreads(
    n_ticks: int,
    start_price: float,
    config: MarketConfig,
    rng: np.random.Generator,
    dt: float,
    initial_spread_ticks: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate rounded bid and ask prices for a batch or one stream block."""

    if n_ticks == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty

    normal_returns = rng.standard_normal(n_ticks)
    log_returns = (
        config.drift_per_second
        - 0.5 * config.volatility_per_sqrt_second**2
    ) * dt
    log_returns = log_returns + (
        config.volatility_per_sqrt_second * np.sqrt(dt) * normal_returns
    )

    if config.jump_intensity_per_second > 0.0 and (
        config.jump_volatility > 0.0 or config.jump_mean != 0.0
    ):
        jump_count = rng.poisson(config.jump_intensity_per_second * dt, n_ticks)
        jump_count_float = jump_count.astype(np.float64, copy=False)
        jump_scale = config.jump_volatility * np.sqrt(jump_count_float)
        jump_returns = rng.normal(
            loc=config.jump_mean * jump_count_float,
            scale=jump_scale,
        )
        log_returns += np.where(jump_count > 0, jump_returns, 0.0)

    mid_prices = start_price * np.exp(np.cumsum(log_returns))
    mid_ticks = np.maximum(
        1,
        np.rint(mid_prices / config.tick_size),
    ).astype(np.int64)

    spread_noise = (
        config.spread_volatility_ticks_per_sqrt_second
        * np.sqrt(dt)
        * rng.standard_normal(n_ticks)
    )
    spread_path = _mean_reverting_path(
        spread_noise,
        baseline=config.baseline_spread_ticks,
        initial=(
            config.baseline_spread_ticks
            if initial_spread_ticks is None
            else initial_spread_ticks
        ),
        reversion=config.spread_reversion_per_second,
        dt=dt,
    )
    spread_ticks = np.maximum(
        np.rint(spread_path).astype(np.int64),
        max(1, int(np.ceil(config.minimum_spread_ticks))),
    )

    # Integer tick arithmetic ensures that prices are rounded and that the
    # quoted spread remains strictly positive even at the minimum spread.
    bid_ticks = np.maximum(1, mid_ticks - spread_ticks // 2)
    ask_ticks = bid_ticks + spread_ticks
    bid_prices = bid_ticks.astype(np.float64) * config.tick_size
    ask_prices = ask_ticks.astype(np.float64) * config.tick_size
    return bid_prices, ask_prices


def _generate_volumes(
    n_ticks: int,
    config: MarketConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if config.volume_distribution == "gamma":
        bid = rng.gamma(config.volume_shape, config.volume_scale, n_ticks)
        ask = rng.gamma(config.volume_shape, config.volume_scale, n_ticks)
    else:
        bid = rng.lognormal(config.volume_log_mean, config.volume_log_sigma, n_ticks)
        ask = rng.lognormal(config.volume_log_mean, config.volume_log_sigma, n_ticks)
    return bid, ask


def _generate_stream_tick(
    timestamp: int,
    current_price: float,
    current_spread_ticks: float,
    config: MarketConfig,
    rng: np.random.Generator,
    dt: float,
) -> tuple[Tick, float, float]:
    """Generate one tick without allocating one-element NumPy arrays."""

    log_return = (
        config.drift_per_second
        - 0.5 * config.volatility_per_sqrt_second**2
    ) * dt
    log_return += config.volatility_per_sqrt_second * sqrt(dt) * float(
        rng.standard_normal()
    )

    if config.jump_intensity_per_second > 0.0 and (
        config.jump_volatility > 0.0 or config.jump_mean != 0.0
    ):
        jump_count = int(rng.poisson(config.jump_intensity_per_second * dt))
        if jump_count:
            log_return += float(
                rng.normal(
                    loc=config.jump_mean * jump_count,
                    scale=config.jump_volatility * sqrt(jump_count),
                )
            )

    next_mid_price = current_price * exp(log_return)
    mid_ticks = max(1, int(round(next_mid_price / config.tick_size)))

    rho = exp(-config.spread_reversion_per_second * dt)
    spread_ticks = config.baseline_spread_ticks + rho * (
        current_spread_ticks - config.baseline_spread_ticks
    )
    spread_ticks += (
        config.spread_volatility_ticks_per_sqrt_second
        * sqrt(dt)
        * float(rng.standard_normal())
    )
    spread_ticks = max(
        float(max(1, int(ceil(config.minimum_spread_ticks)))),
        spread_ticks,
    )
    spread_ticks_int = max(1, int(round(spread_ticks)))

    bid_ticks = max(1, mid_ticks - spread_ticks_int // 2)
    ask_ticks = bid_ticks + spread_ticks_int
    bid_price = bid_ticks * config.tick_size
    ask_price = ask_ticks * config.tick_size

    if config.volume_distribution == "gamma":
        bid_size = float(rng.gamma(config.volume_shape, config.volume_scale))
        ask_size = float(rng.gamma(config.volume_shape, config.volume_scale))
    else:
        bid_size = float(rng.lognormal(config.volume_log_mean, config.volume_log_sigma))
        ask_size = float(rng.lognormal(config.volume_log_mean, config.volume_log_sigma))

    return (
        (timestamp, bid_price, ask_price, bid_size, ask_size),
        (bid_price + ask_price) * 0.5,
        float(spread_ticks_int),
    )


class MarketGenerator:
    """Synthetic tick generator with reusable configuration and RNG state."""

    __slots__ = ("config", "rng")

    def __init__(
        self,
        config: MarketConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config if config is not None else MarketConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_batch(self, n_ticks: int, start_price: float) -> np.ndarray:
        """Generate ``n_ticks`` into one contiguous structured NumPy array."""

        if isinstance(n_ticks, bool) or not isinstance(n_ticks, (int, np.integer)):
            raise TypeError("n_ticks must be an integer")
        if n_ticks < 0:
            raise ValueError("n_ticks must not be negative")
        start_price = _validate_start_price(start_price)

        data = np.empty(int(n_ticks), dtype=TICK_DTYPE)
        if n_ticks == 0:
            return data

        dt = self.config.sample_interval_seconds
        bid_prices, ask_prices = _generate_prices_and_spreads(
            int(n_ticks), start_price, self.config, self.rng, dt
        )
        bid_sizes, ask_sizes = _generate_volumes(int(n_ticks), self.config, self.rng)

        interval_ns = max(1, int(round(dt * 1_000_000_000.0)))
        offsets = np.arange(int(n_ticks), dtype=np.int64) * np.int64(interval_ns)
        data["timestamp_ns"] = np.int64(time_ns()) + offsets
        data["bid_price"] = bid_prices
        data["ask_price"] = ask_prices
        data["bid_size"] = bid_sizes
        data["ask_size"] = ask_sizes
        return data

    def stream_ticks(
        self,
        target_rate: int = 50_000,
        start_price: float = 100.0,
    ) -> Iterator[Tick]:
        """Yield individual ticks at approximately ``target_rate`` per second.

        The final part of each interval is spin-waited using
        :func:`perf_counter_ns` for accurate pacing.  This intentionally
        consumes one CPU core while the stream is active.
        """

        if isinstance(target_rate, bool) or not isinstance(
            target_rate, (int, np.integer)
        ):
            raise TypeError("target_rate must be an integer")
        if target_rate <= 0:
            raise ValueError("target_rate must be greater than zero")
        start_price = _validate_start_price(start_price)

        dt = 1.0 / int(target_rate)
        interval_ns = max(1, int(round(1_000_000_000.0 / int(target_rate))))
        current_price = start_price
        current_spread_ticks = self.config.baseline_spread_ticks
        deadline = perf_counter_ns()

        while True:
            _wait_until(deadline)
            timestamp = time_ns()

            tick, current_price, current_spread_ticks = _generate_stream_tick(
                timestamp,
                current_price,
                current_spread_ticks,
                self.config,
                self.rng,
                dt,
            )
            yield tick
            deadline += interval_ns


def _wait_until(deadline_ns: int) -> None:
    """Sleep coarsely, then spin, until a perf-counter deadline."""

    while True:
        remaining = deadline_ns - perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            sleep((remaining - 100_000) / 1_000_000_000.0)
        else:
            while perf_counter_ns() < deadline_ns:
                pass
            return


def generate_batch(
    n_ticks: int,
    start_price: float,
    *,
    config: MarketConfig | None = None,
    tick_size: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a structured batch using an optional configuration."""

    resolved = _resolve_config(config, tick_size)
    return MarketGenerator(resolved, rng).generate_batch(n_ticks, start_price)


def stream_ticks(
    target_rate: int = 50_000,
    *,
    start_price: float = 100.0,
    config: MarketConfig | None = None,
    tick_size: float | None = None,
    rng: np.random.Generator | None = None,
) -> Iterator[Tick]:
    """Yield tuple-form ticks at a paced target rate."""

    resolved = _resolve_config(config, tick_size)
    yield from MarketGenerator(resolved, rng).stream_ticks(target_rate, start_price)


def _benchmark() -> None:
    """Run the requested five-second streaming and batch verification."""

    generator = MarketGenerator(rng=np.random.default_rng(7))
    stream = generator.stream_ticks(target_rate=50_000)
    duration_ns = 5_000_000_000
    started_ns = perf_counter_ns()
    deadline_ns = started_ns + duration_ns
    count = 0
    while perf_counter_ns() < deadline_ns:
        next(stream)
        count += 1
    elapsed_seconds = (perf_counter_ns() - started_ns) / 1_000_000_000.0
    stream.close()

    batch_started_ns = perf_counter_ns()
    data = generator.generate_batch(1_000_000, start_price=100.0)
    batch_elapsed_seconds = (perf_counter_ns() - batch_started_ns) / 1_000_000_000.0

    assert data.dtype == TICK_DTYPE
    assert np.all(data["ask_price"] > data["bid_price"])

    print(
        f"stream: {count:,} ticks in {elapsed_seconds:.6f}s "
        f"({count / elapsed_seconds:,.0f} ticks/sec)"
    )
    print(
        f"batch:  {len(data):,} ticks in {batch_elapsed_seconds:.6f}s "
        f"({len(data) / batch_elapsed_seconds:,.0f} ticks/sec)"
    )
    print(f"dtype:   {data.dtype == TICK_DTYPE}")
    print("spread:  ask_price > bid_price verified")


if __name__ == "__main__":
    _benchmark()
