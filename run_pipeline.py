"""Run the AetherFlow engine and print one live OHLCV snapshot per second."""

from __future__ import annotations

import time

from engine import MasterEngine


RUNTIME_SECONDS = 5
STREAM_RATE = 50_000
OHLCV_TICKS = 10_000


def _print_latest_bar(engine: MasterEngine) -> None:
    ticks_written = int(engine.master_rb.write_head)
    bars = engine.analytics.get_ohlcv(
        "PRIMARY",
        timeframe="1s",
        last_n_ticks=OHLCV_TICKS,
    )

    if bars.empty:
        print(f"ticks={ticks_written:,} | no completed 1s bar yet", flush=True)
        return

    latest = bars.iloc[-1]
    timestamp = bars.index[-1]
    print(
        f"ticks={ticks_written:,} | timestamp={timestamp} | "
        f"open={float(latest['open']):.6f} "
        f"high={float(latest['high']):.6f} "
        f"low={float(latest['low']):.6f} "
        f"close={float(latest['close']):.6f} "
        f"volume={float(latest['volume']):.6f} "
        f"tick_count={int(latest['tick_count'])}",
        flush=True,
    )


def main() -> None:
    engine = MasterEngine(n_workers=2)
    try:
        engine.start(stream_rate=STREAM_RATE)
        for _ in range(RUNTIME_SECONDS):
            time.sleep(1.0)
            _print_latest_bar(engine)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received; stopping AetherFlow.", flush=True)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
