"""
AetherFlow vs Standard Python IPC Benchmark (Pure Transmission Speed)
Compares tick-by-tick Queue serialization against Zero-Copy Shared Memory streaming.
"""

import time
import multiprocessing as mp
import numpy as np
from protocol import TICK_DTYPE
from ring_buffer import SharedMemoryRingBuffer
from market_generator import generate_batch

N_BENCHMARK_TICKS = 100_000


# --- 1. Baseline: multiprocessing.Queue (Tick-by-Tick) ---
def queue_producer(q: mp.Queue, ticks: np.ndarray, ready_event: mp.Event, start_event: mp.Event):
    ready_event.set()
    start_event.wait()
    # Sends individual tick objects through pickle serialization
    for i in range(len(ticks)):
        q.put(ticks[i])
    q.put(None)


def queue_consumer(q: mp.Queue):
    while True:
        chunk = q.get()
        if chunk is None:
            break


def run_queue_benchmark(ticks: np.ndarray) -> float:
    q = mp.Queue()
    ready = mp.Event()
    start = mp.Event()

    prod = mp.Process(target=queue_producer, args=(q, ticks, ready, start))
    cons = mp.Process(target=queue_consumer, args=(q,))

    cons.start()
    prod.start()
    ready.wait()  # Wait until producer process is fully booted

    t0 = time.perf_counter()
    start.set()   # Release the producer
    prod.join()
    cons.join()
    elapsed = time.perf_counter() - t0

    return len(ticks) / elapsed


# --- 2. AetherFlow: Shared Memory (Zero-Copy Streaming) ---
def shm_producer(shm_name: str, ticks: np.ndarray, ready_event: mp.Event, start_event: mp.Event):
    rb = SharedMemoryRingBuffer(name=shm_name, capacity=131_072, create=False)
    ready_event.set()
    start_event.wait()
    # Streams market feed directly into shared memory with zero copy
    chunk_size = 1_000
    for i in range(0, len(ticks), chunk_size):
        rb.write_batch(ticks[i : i + chunk_size])
    rb.close()


def run_shm_benchmark(ticks: np.ndarray) -> float:
    shm_name = "aetherflow_bench_shm"
    master_rb = SharedMemoryRingBuffer(name=shm_name, capacity=131_072, create=True)
    ready = mp.Event()
    start = mp.Event()

    try:
        prod = mp.Process(target=shm_producer, args=(shm_name, ticks, ready, start))
        prod.start()
        ready.wait()  # Wait until producer process is fully booted

        t0 = time.perf_counter()
        start.set()   # Release the producer

        read_ticks = 0
        while read_ticks < len(ticks):
            read_ticks = master_rb.write_head

        prod.join()
        elapsed = time.perf_counter() - t0
        return len(ticks) / elapsed
    finally:
        master_rb.close()
        master_rb.unlink()


def main():
    print(f"Generating {N_BENCHMARK_TICKS:,} synthetic ticks for head-to-head benchmark...")
    ticks = generate_batch(N_BENCHMARK_TICKS)

    print("\n[1/2] Benchmarking Standard multiprocessing.Queue (Tick-by-Tick Pickle)...")
    queue_rate = run_queue_benchmark(ticks)
    print(f"  Queue Throughput:      {queue_rate:,.0f} ticks/sec")

    print("\n[2/2] Benchmarking AetherFlow Zero-Copy Shared Memory Streaming...")
    shm_rate = run_shm_benchmark(ticks)
    print(f"  AetherFlow Throughput: {shm_rate:,.0f} ticks/sec")

    speedup = shm_rate / queue_rate if queue_rate > 0 else 0
    print("\n==================================================")
    print(f"  AETHERFLOW SPEEDUP: {speedup:.1f}x FASTER THAN QUEUE")
    print("==================================================")


if __name__ == "__main__":
    main()