## Day 1: Synthetic Data Generator & Protocol Design (4 Hours)

Simple description: 

Goal: Create realistic L2 tick and trade streams without needing an active
live exchange account.

Tasks:
Define your binary tick structure using a custom NumPy structured dtype:
TICK_DTYPE = np.dtype([
('timestamp_ns', np.int64),
('bid_price', np.float64),
('ask_price', np.float64),
('bid_size', np.float64),
('ask_size', np.float64)
])

Build a synthetic exchange feed using a Poisson jump process or
Geometric Brownian Motion that spits out 50,000+ ticks/sec. (Can be
largely assisted by an LLM).



### Manual: protocol.py, verify_layout.py
### Vibecode: market_generator.py



┌────────────────────────────────────────────────────────┐
│               VIBECODE (SEND TO CODEX)                 │
├────────────────────────────────────────────────────────┤
│ • Stochastic price path math (GBM / Poisson jumps)     │
│ • Realistic queue dynamics (bid/ask sizing distributions)│
│ • High-frequency loop mechanics & tick-rate pacing     │
│ • Fast vectorized synthetic batch generator            │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                MUST CODE MANUALLY                      │
├────────────────────────────────────────────────────────┤
│ • Exact byte-level layout of `TICK_DTYPE` & byte alignment│
│ • Memory footprint verification (`itemsize`, byte offsets)│
│ • Struct packing sanity checks (`tobytes()` / buffer read)│
│ • Nanosecond clock semantics (`time.perf_counter_ns`) │
└────────────────────────────────────────────────────────┘

