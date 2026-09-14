"""
AetherFlow Microstructure Feature Engine
Numba-accelerated calculations for OFI, Realized Volatility, and EWMA
operating directly over circular buffer slices with nogil=True.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True, cache=True)
def compute_ofi_tick(
    bid_p_curr: float, bid_p_prev: float,
    bid_s_curr: float, bid_s_prev: float,
    ask_p_curr: float, ask_p_prev: float,
    ask_s_curr: float, ask_s_prev: float
) -> float:
    """
    Calculates single-tick Order Flow Imbalance (OFI) according to Cont-Kukanov-Stoikov (2014).
    Compiled to pure machine instructions without GIL.
    """
    # Bid contribution
    if bid_p_curr > bid_p_prev:
        delta_v_bid = bid_s_curr
    elif bid_p_curr == bid_p_prev:
        delta_v_bid = bid_s_curr - bid_s_prev
    else:
        delta_v_bid = -bid_s_prev

    # Ask contribution
    if ask_p_curr < ask_p_prev:
        delta_v_ask = ask_s_curr
    elif ask_p_curr == ask_p_prev:
        delta_v_ask = ask_s_curr - ask_s_prev
    else:
        delta_v_ask = -ask_s_prev

    return delta_v_bid - delta_v_ask


@njit(fastmath=True, nogil=True, cache=True)
def compute_rolling_realized_vol(mid_prices: np.ndarray) -> float:
    """
    Computes realized volatility over a sliding window of mid-prices.
    Uses log-returns: r_t = ln(P_t / P_{t-1}).
    """
    n = len(mid_prices)
    if n < 2:
        return 0.0

    sum_sq_returns = 0.0
    for i in range(1, n):
        prev = mid_prices[i - 1]
        curr = mid_prices[i]
        if prev > 0.0 and curr > 0.0:
            ret = np.log(curr / prev)
            sum_sq_returns += ret * ret

    return np.sqrt(sum_sq_returns)


@njit(fastmath=True, nogil=True, cache=True)
def compute_rolling_ewma(prices: np.ndarray, alpha: float) -> float:
    """
    Iteratively calculates the EWMA over an array of prices using decay factor alpha.
    Recursive formulation: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}.
    """
    n = len(prices)
    if n == 0:
        return 0.0

    ewma = prices[0]
    for i in range(1, n):
        ewma = alpha * prices[i] + (1.0 - alpha) * ewma

    return ewma


@njit(fastmath=True, nogil=True, cache=True)
def compute_window_features(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    alpha: float
) -> tuple:
    """
    Fused kernel: Computes OFI series, Realized Volatility, and EWMA in a single pass.
    Maximizes L1 cache locality by walking arrays simultaneously.
    """
    n = len(bid_prices)
    if n < 2:
        return 0.0, 0.0, 0.0

    # 1. Rolling OFI sum
    total_ofi = 0.0
    for i in range(1, n):
        total_ofi += compute_ofi_tick(
            bid_prices[i], bid_prices[i - 1],
            bid_sizes[i], bid_sizes[i - 1],
            ask_prices[i], ask_prices[i - 1],
            ask_sizes[i], ask_sizes[i - 1]
        )

    # 2. Mid prices for Vol and EWMA
    mid_prices = 0.5 * (bid_prices + ask_prices)
    realized_vol = compute_rolling_realized_vol(mid_prices)
    ewma_price = compute_rolling_ewma(mid_prices, alpha)

    return total_ofi, realized_vol, ewma_price