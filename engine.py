"""
AetherFlow Master Process Orchestrator
Coordinates Feed Ingestion, Parallel Signal Workers, and Analytics Slicing
with deterministic POSIX lifecycle management and SIGINT handling.
"""

import os
import signal
import sys
import time
import multiprocessing as mp
from typing import List

import numpy as np
from protocol import TICK_DTYPE
from ring_buffer import SharedMemoryRingBuffer
from market_generator import generate_batch
from features import compute_window_features
from analytics import AnalyticsEngine


SHM_NAME_DEFAULT = "aetherflow_master_ring"
BUFFER_CAPACITY_DEFAULT = 1_048_576  # 2^20 slots (~40 MB)
FEATURE_WINDOW_SIZE = 1_000
EWMA_ALPHA = 0.05


def producer_worker(shm_name: str, capacity: int, stop_event: mp.Event, target_rate: int = 50_000):
    """
    Producer Process: Ingests market data ticks directly into shared memory.
    Runs at target_rate ticks/second.
    """
    # Ignore SIGINT in child workers; master handles shutdown
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    rb = SharedMemoryRingBuffer(name=shm_name, capacity=capacity, create=False)
    batch_chunk = 1_000
    sleep_interval = batch_chunk / target_rate  # e.g., 20ms for 1000 ticks at 50k/sec

    print(f"[Producer PID {os.getpid()}] Streaming at target {target_rate:,} ticks/sec...")

    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            batch = generate_batch(batch_chunk)
            rb.write_batch(batch)

            elapsed = time.perf_counter() - t0
            remaining = sleep_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    finally:
        rb.close()
        print(f"[Producer PID {os.getpid()}] Exited cleanly.")


def compute_worker(worker_id: int, shm_name: str, capacity: int, stop_event: mp.Event):
    """
    Consumer Compute Process: Continuously evaluates microstructure features
    on sliding windows without touching Python's GIL.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    rb = SharedMemoryRingBuffer(name=shm_name, capacity=capacity, create=False)
    last_processed_seq = 0
    eval_count = 0

    print(f"[Worker-{worker_id} PID {os.getpid()}] Compute engine active.")

    try:
        while not stop_event.is_set():
            current_head = rb.write_head

            # Evaluate when at least 100 new ticks arrive and enough history exists
            if current_head - last_processed_seq >= 100 and current_head >= FEATURE_WINDOW_SIZE:
                window = rb.read_last_n(FEATURE_WINDOW_SIZE)
                last_processed_seq = current_head

                # Extract contiguous views
                bid_p = np.ascontiguousarray(window['bid_price'])
                ask_p = np.ascontiguousarray(window['ask_price'])
                bid_s = np.ascontiguousarray(window['bid_size'])
                ask_s = np.ascontiguousarray(window['ask_size'])

                # Fused Numba computation (nogil=True)
                ofi, vol, ewma = compute_window_features(bid_p, ask_p, bid_s, ask_s, EWMA_ALPHA)
                eval_count += 1
            else:
                # Avoid CPU spin-lock starvation
                time.sleep(0.0005)
    finally:
        rb.close()
        print(f"[Worker-{worker_id} PID {os.getpid()}] Processed {eval_count:,} feature windows. Exited.")


class MasterEngine:
    def __init__(self, shm_name: str = SHM_NAME_DEFAULT, capacity: int = BUFFER_CAPACITY_DEFAULT, n_workers: int = 2):
        self.shm_name = shm_name
        self.capacity = capacity
        self.n_workers = n_workers

        # Initialize OS Shared Memory Block
        self.master_rb = SharedMemoryRingBuffer(name=self.shm_name, capacity=self.capacity, create=True)
        self.stop_event = mp.Event()
        self.processes: List[mp.Process] = []

        # Connect Analytics Layer
        self.analytics = AnalyticsEngine({"PRIMARY": self.master_rb})

        # Register termination signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n[Master] Termination signal caught. Initiating graceful shutdown...")
        self.stop()
        sys.exit(0)

    def start(self, stream_rate: int = 50_000):
        """Spawns all producer and compute worker child processes."""
        print(f"[Master PID {os.getpid()}] Initializing AetherFlow Engine...")

        # 1. Spawn Producer Process
        prod = mp.Process(
            target=producer_worker,
            args=(self.shm_name, self.capacity, self.stop_event, stream_rate),
            name="AetherProducer"
        )
        self.processes.append(prod)

        # 2. Spawn Parallel Signal Workers
        for i in range(self.n_workers):
            worker = mp.Process(
                target=compute_worker,
                args=(i + 1, self.shm_name, self.capacity, self.stop_event),
                name=f"AetherWorker-{i+1}"
            )
            self.processes.append(worker)

        # Start all processes
        for p in self.processes:
            p.daemon = True
            p.start()

        print(f"[Master] All {len(self.processes)} sub-processes spawned successfully.\n")

    def stop(self):
        """Stops child processes and unlinks shared memory to prevent leaks."""
        self.stop_event.set()

        for p in self.processes:
            p.join(timeout=1.5)
            if p.is_alive():
                p.terminate()

        # Clean up shared memory
        self.master_rb.close()
        self.master_rb.unlink()
        print("[Master] Shared memory unlinked. Engine shutdown complete.")