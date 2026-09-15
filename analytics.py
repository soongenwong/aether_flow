"""
AetherFlow Analytics Layer
Translates shared-memory circular buffer views into multi-timeframe OHLCV bars
and rolling multi-asset correlation matrices via Pandas.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ring_buffer import SharedMemoryRingBuffer


class AnalyticsEngine:
    def __init__(self, buffers: Optional[Dict[str, SharedMemoryRingBuffer]] = None):
        """
        Parameters:
        -----------
        buffers : Dict[str, SharedMemoryRingBuffer]
            Map of symbol names to their respective shared memory ring buffers.
        """
        self.buffers: Dict[str, SharedMemoryRingBuffer] = buffers or {}

    def register_buffer(self, symbol: str, buffer: SharedMemoryRingBuffer) -> None:
        """Register a shared memory buffer for a given symbol."""
        self.buffers[symbol] = buffer

    def get_raw_dataframe(self, symbol: str, last_n: Optional[int] = None) -> pd.DataFrame:
        """
        Extracts a slice from the ring buffer into a Pandas DataFrame.
        Constructs DataFrame directly from the structured array.
        """
        if symbol not in self.buffers:
            raise KeyError(f"Symbol '{symbol}' not registered in AnalyticsEngine.")

        rb = self.buffers[symbol]
        n = last_n if last_n is not None else min(rb.write_head, rb.capacity)
        if n == 0:
            return pd.DataFrame(columns=['timestamp_ns', 'bid_price', 'ask_price', 'bid_size', 'ask_size', 'mid_price'])

        # 1. Read slice from ring buffer
        raw_ticks = rb.read_last_n(n)

        # 2. Construct DataFrame from structured NumPy array
        df = pd.DataFrame(raw_ticks)
        
        # 3. Vectorized derived columns
        df['timestamp'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
        df['mid_price'] = 0.5 * (df['bid_price'] + df['ask_price'])
        df['total_size'] = df['bid_size'] + df['ask_size']
        df.set_index('timestamp', inplace=True)
        return df

    def get_ohlcv(self, symbol: str, timeframe: str = "1s", last_n_ticks: Optional[int] = None) -> pd.DataFrame:
        """
        Resamples live tick stream into Open, High, Low, Close, Volume bars.
        
        Parameters:
        -----------
        timeframe : str
            Pandas offset string: '100ms', '1s', '5s', '1min', etc.
        """
        df = self.get_raw_dataframe(symbol, last_n=last_n_ticks)
        if df.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'tick_count'])

        # Resample mid-price to OHLC
        ohlc = df['mid_price'].resample(timeframe).ohlc()
        
        # Resample volume and activity
        volume = df['total_size'].resample(timeframe).sum()
        tick_count = df['mid_price'].resample(timeframe).count()

        bars = ohlc.copy()
        bars['volume'] = volume
        bars['tick_count'] = tick_count

        # Drop periods with zero ticks
        return bars.dropna(subset=['close'])

    def compute_rolling_correlation(
        self,
        symbols: List[str],
        timeframe: str = "1s",
        rolling_window: int = 30,
        last_n_ticks: int = 50_000
    ) -> pd.DataFrame:
        """
        Synchronizes multiple tick streams onto a common time grid,
        resamples to bars, and computes the rolling pairwise correlation matrix.
        """
        close_series = {}

        for sym in symbols:
            bars = self.get_ohlcv(sym, timeframe=timeframe, last_n_ticks=last_n_ticks)
            if not bars.empty:
                close_series[sym] = bars['close']

        if len(close_series) < 2:
            raise ValueError("Need at least 2 symbols with active bars to compute correlation.")

        # Align on common timestamp index, forward-fill missing seconds
        price_matrix = pd.DataFrame(close_series).ffill().dropna()

        # Compute percentage returns
        returns = price_matrix.pct_change().dropna()

        if len(returns) < rolling_window:
            # Fall back to full available return correlation if history is shorter than window
            return returns.corr()

        # Compute pairwise correlation over the most recent rolling window
        return returns.tail(rolling_window).corr()