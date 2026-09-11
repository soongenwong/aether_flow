1. Streaming Market Data & Feature Engine
Score: 96 / 100
What it is: Ingests live tick data feeds, passes updates via zero-copy shared memory (multiprocessing.shared_memory) to isolated worker pools, and uses contiguous NumPy circular buffers and Numba JIT to compute rolling microstructural metrics (Order Flow Imbalance, realized volatility, EWMA). Exposes a Pandas aggregation layer for multi-timeframe OHLCV bars.
Why it wins: Tackles the exact problems hedge fund data platform teams face daily: concurrency without GIL lockups, memory locality, and high throughput.



2. Deep Dive: What the Project Actually Does
In quantitative trading, strategies need real-time streaming signals calculated continuously as trades and quotes hit the network. If your code runs slow or stalls during a market burst, trades execute late, and the fund loses money.
This project solves three classic engineering bottlenecks in Python:
The Concurrency Bottleneck (GIL): Standard Python multi-threading runs on one core. Python's multiprocessing.Queue copies and serializes every object using pickle, which destroys throughput. You fix this using POSIX shared memory (multiprocessing.shared_memory), passing binary tick data with zero data copying.
The Memory Allocation Bottleneck: Continuously appending data to Python lists or resizing arrays triggers Python’s garbage collector and causes latency spikes. You fix this using fixed-size circular ring buffers pre-allocated in shared memory as contiguous C-ordered NumPy arrays.
The Calculation Bottleneck: Computing rolling statistics on a stream requires loops over past ticks. You fix this using Numba JIT (@njit(nogil=True)) to run C-speed compiled math over your circular buffers.
The Downstream Interface: Downstream quant researchers prefer clean APIs. You write an analytical layer using Pandas that slices these live circular buffers into multi-timeframe OHLCV bars and rolling correlation matrices on demand.



Step-by-Step 1-Week Project Timeline (Total: ~30–35 Hours)
Assuming you have 1 week (working 4–6 hours per day), here is how to break down
the build:
Day 1: Data Structs & Ingestion  ──>  Day 2: Shared Memory & Ring Buffer
│
Day 4: Pandas Analytical API    <──  Day 3: NumPy + Numba Features
│
Day 5: Concurrency & Sync        ──>  Day 6: Benchmarks & Profiling  ──>  Day 7: CV Polish