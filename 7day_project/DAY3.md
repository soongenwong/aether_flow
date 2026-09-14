## Day 3: NumPy Core & Numba JIT Feature Compute (6 Hours) — [CRITICAL]

Goal: Calculate live financial microstructure signals at compiled C speeds.

Tasks:
Write Numba functions decorated with @njit(fastmath=True, nogil=True)
that accept your circular buffer array and the current head index.

Implement Order Flow Imbalance (OFI), Rolling Realized Volatility, and
Rolling EWMA.

Verify that Numba compiles cleanly without falling back to object mode
(nopython=True).



### Manual: features.py
### Vibecode: test_features.py



┌────────────────────────────────────────────────────────┐
│               CAN BE VIBECODED (CODEX)                 │
├────────────────────────────────────────────────────────┤
│ • Mathematical formulas for OFI, EWMA, & Volatility    │
│ • Numerical stability edge cases (div-by-zero, log(0)) │
│ • Benchmark harness measuring microsecond latencies    │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                  MUST CODE MANUALLY                    │
├────────────────────────────────────────────────────────┤
│ • Numba signature optimization & `@njit(nogil=True)`   │
│ • Ring-buffer circular index extraction inside Numba   │
│ • Ensuring zero object-mode fallback (pure C types)    │
│ • Passing raw structured arrays or strided views       │
└────────────────────────────────────────────────────────┘

