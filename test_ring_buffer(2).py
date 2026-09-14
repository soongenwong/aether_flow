"""Standalone verification and throughput harness for the shared tick ring."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty
from time import perf_counter_ns, sleep
from traceback import format_exc
from typing import Any

import numpy as np

from market_generator import generate_batch
from protocol import TICK_DTYPE
from ring_buffer import SharedMemoryRingBuffer


SHARED_MEMORY_NAME = "aetherflow_test"
CAPACITY = 1_048_576
TICK_COUNT = 1_000_000
START_PRICE = 100.0
RNG_SEED = 7
UNIT_CAPACITY = 1_024
UNIT_NAME = "aetherflow_unit_test"
POLL_TIMEOUT_SECONDS = 30.0


def _digest_ticks(ticks: np.ndarray) -> str:
    """Return a compact digest of the exact structured-array bytes."""

    contiguous = np.ascontiguousarray(ticks)
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()


def _assert_same_tick(actual: np.void, expected: np.void) -> None:
    for field in TICK_DTYPE.names or ():
        assert actual[field] == expected[field], (
            f"field {field!r} differs: {actual[field]!r} != {expected[field]!r}"
        )


def run_single_process_test() -> None:
    """Exercise bitwise wrap-around with individual writes."""

    total_ticks = UNIT_CAPACITY + 500
    expected = np.empty(total_ticks, dtype=TICK_DTYPE)
    sequence = np.arange(total_ticks, dtype=np.int64)
    expected["timestamp_ns"] = sequence
    expected["bid_price"] = 100.0 + sequence.astype(np.float64) * 0.01
    expected["ask_price"] = expected["bid_price"] + 0.01
    expected["bid_size"] = 10.0 + sequence.astype(np.float64)
    expected["ask_size"] = 20.0 + sequence.astype(np.float64)

    ring: SharedMemoryRingBuffer | None = None
    try:
        ring = SharedMemoryRingBuffer(
            name=UNIT_NAME,
            capacity=UNIT_CAPACITY,
            create=True,
        )
        for tick in expected:
            ring.write_tick(tick)

        assert ring.write_head == total_ticks
        _assert_same_tick(ring.read_latest(), expected[-1])
        assert np.array_equal(ring.read_last_n(500), expected[-500:])
        assert np.array_equal(
            ring.read_last_n(UNIT_CAPACITY), expected[-UNIT_CAPACITY:]
        )
    finally:
        if ring is not None:
            try:
                ring.close()
            finally:
                ring.unlink()


def _producer_worker(
    ready: Any,
    consumer_done: Any,
    result_queue: Any,
) -> None:
    """Create, populate, and own the shared-memory segment."""

    ring: SharedMemoryRingBuffer | None = None
    try:
        ring = SharedMemoryRingBuffer(
            name=SHARED_MEMORY_NAME,
            capacity=CAPACITY,
            create=True,
        )
        ready.set()

        batch = generate_batch(
            TICK_COUNT,
            START_PRICE,
            rng=np.random.default_rng(RNG_SEED),
        )
        assert batch.dtype == TICK_DTYPE
        digest = _digest_ticks(batch)

        write_started_ns = perf_counter_ns()
        ring.write_batch(batch)
        write_elapsed_seconds = (
            perf_counter_ns() - write_started_ns
        ) / 1_000_000_000.0

        result_queue.put(
            {
                "ok": True,
                "digest": digest,
                "write_elapsed_seconds": write_elapsed_seconds,
                "write_head": ring.write_head,
            }
        )
        consumer_done.wait(POLL_TIMEOUT_SECONDS)
    except BaseException:
        result_queue.put({"ok": False, "error": format_exc()})
        ready.set()
        raise
    finally:
        if ring is not None:
            try:
                ring.close()
            finally:
                ring.unlink()


def _force_unlink(name: str) -> None:
    """Remove a named segment left behind by an abnormal child exit."""

    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    finally:
        shm.close()


def _get_child_failure(result_queue: Any) -> str | None:
    try:
        result = result_queue.get_nowait()
    except Empty:
        return None
    if result.get("ok"):
        return None
    return str(result.get("error", "producer failed without an error message"))


def run_multi_process_test() -> None:
    """Verify exact cross-process transfer and report producer/consumer rates."""

    context = mp.get_context("spawn")
    ready = context.Event()
    consumer_done = context.Event()
    result_queue = context.Queue()
    producer = context.Process(
        target=_producer_worker,
        args=(ready, consumer_done, result_queue),
        name="aetherflow-producer",
    )

    consumer: SharedMemoryRingBuffer | None = None
    try:
        producer.start()
        if not ready.wait(POLL_TIMEOUT_SECONDS):
            raise TimeoutError("producer did not initialize shared memory")

        consumer = SharedMemoryRingBuffer(
            name=SHARED_MEMORY_NAME,
            capacity=CAPACITY,
            create=False,
        )

        last_head = 0
        progression: list[int] = []
        observe_started_ns = perf_counter_ns()
        observe_deadline_ns = observe_started_ns + int(
            POLL_TIMEOUT_SECONDS * 1_000_000_000
        )

        while last_head < TICK_COUNT:
            head = consumer.write_head
            assert head >= last_head, (
                f"write_head moved backwards: {head} < {last_head}"
            )
            assert head <= TICK_COUNT, (
                f"write_head exceeded expected count: {head} > {TICK_COUNT}"
            )
            if head != last_head:
                progression.append(head)
                last_head = head

            if perf_counter_ns() >= observe_deadline_ns:
                failure = _get_child_failure(result_queue)
                detail = f"\n{failure}" if failure else ""
                raise TimeoutError(f"producer did not finish{detail}")
            if last_head < TICK_COUNT:
                sleep(0.0005)

        observe_elapsed_seconds = (
            perf_counter_ns() - observe_started_ns
        ) / 1_000_000_000.0
        assert last_head == TICK_COUNT
        assert progression, "consumer observed no write_head progression"
        assert progression[-1] == TICK_COUNT

        read_started_ns = perf_counter_ns()
        received = consumer.read_last_n(TICK_COUNT)
        latest = consumer.read_latest()
        read_elapsed_seconds = (
            perf_counter_ns() - read_started_ns
        ) / 1_000_000_000.0

        assert received.dtype == TICK_DTYPE
        assert len(received) == TICK_COUNT
        assert np.all(received["ask_price"] > received["bid_price"])
        assert np.all(np.diff(received["timestamp_ns"]) > 0)
        _assert_same_tick(latest, received[-1])

        result = result_queue.get(timeout=POLL_TIMEOUT_SECONDS)
        if not result.get("ok"):
            raise AssertionError(result.get("error", "producer failed"))
        assert result["write_head"] == TICK_COUNT
        assert _digest_ticks(received) == result["digest"], (
            "shared-memory payload differs from the producer payload"
        )

        write_elapsed_seconds = float(result["write_elapsed_seconds"])
        print("single-process wrap-around: PASS")
        print(
            "multi-process transfer: PASS "
            f"({len(progression)} write_head updates observed)"
        )
        print(
            f"write throughput: {TICK_COUNT / write_elapsed_seconds:,.0f} ticks/sec"
        )
        print(
            f"read throughput:  {TICK_COUNT / read_elapsed_seconds:,.0f} ticks/sec"
        )
        print(
            f"end-to-end observed: {TICK_COUNT / observe_elapsed_seconds:,.0f} ticks/sec"
        )
    finally:
        consumer_done.set()
        if consumer is not None:
            try:
                consumer.close()
            finally:
                consumer.unlink()

        producer.join(POLL_TIMEOUT_SECONDS)
        if producer.is_alive():
            producer.terminate()
            producer.join()
        _force_unlink(SHARED_MEMORY_NAME)


def main() -> None:
    run_single_process_test()
    run_multi_process_test()


if __name__ == "__main__":
    main()

