## Day 3: NumPy Core & Numba JIT Feature Compute (6 Hours) — [CRITICAL]

Goal: Calculate live financial microstructure signals at compiled C speeds.

Tasks:
Write Numba functions decorated with @njit(fastmath=True, nogil=True)
that accept your circular buffer array and the current head index.

Implement Order Flow Imbalance (OFI), Rolling Realized Volatility, and
Rolling EWMA.

Verify that Numba compiles cleanly without falling back to object mode
(nopython=True).