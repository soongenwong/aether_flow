Day 2: Shared Memory & Circular Buffer Engine (6 Hours) — [CRITICAL]
Goal: Enable two completely isolated Python processes to communicate without
sockets or queues.
Tasks:
Implement multiprocessing.shared_memory.SharedMemory.
Write the CircularBuffer class manually. Ensure it pre-allocates an
array of shape (N,) with TICK_DTYPE on the shared buffer.
Implement read/write pointers and atomic synchronization (using a shared
multiprocessing.Value or RawArray flag).