## Day 2: Shared Memory & Circular Buffer Engine (6 Hours) — [CRITICAL]

Simple description: 

Goal: Enable two completely isolated Python processes to communicate without
sockets or queues.

Tasks:
Implement multiprocessing.shared_memory.SharedMemory.

Write the CircularBuffer class manually. Ensure it pre-allocates an
array of shape (N,) with TICK_DTYPE on the shared buffer.

Implement read/write pointers and atomic synchronization (using a shared
multiprocessing.Value or RawArray flag).



### Manual: ring_buffer.py (must understand every line fully and how the big picture works)
### Vibecode: test_ring_buffer.py

┌────────────────────────────────────────────────────────┐
│               CAN BE VIBECODED (CODEX)                 │
├────────────────────────────────────────────────────────┤
│ • Unit test harness asserting cross-process writes     │
│ • Benchmark script comparing Queue vs Shared Memory    │
│ • Multi-process producer/consumer test runner          │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                  MUST CODE MANUALLY                    │
├────────────────────────────────────────────────────────┤
│ • `SharedMemoryRingBuffer` class implementation        │
│ • Power-of-2 bitwise wrapping math (`& (capacity - 1)`)│
│ • Direct `np.ndarray(..., buffer=...)` offset mapping  │
│ • Clean initialization vs. attach-existing logic       │
│ • Explicit OS resource cleanup (`close()` & `unlink()`)│
└────────────────────────────────────────────────────────┘

