"""
AetherFlow Shared Memory Circular Ring Buffer
Provides zero-copy, inter-process communication using OS shared memory
and bitwise-wrapped contiguous C-arrays.
"""

import math
from multiprocessing import shared_memory
import numpy as np
from protocol import TICK_DTYPE, TICK_ITEMSIZE

# Header Layout:
# Offset 0:  write_head (int64, 8 bytes) -> monotonic write sequence index
# Offset 8:  is_active  (int64, 8 bytes) -> 1 if running, 0 if shutting down
# Offset 16..63: Reserved cache-line padding (ensures data buffer starts on 64-byte boundary)
HEADER_SIZE_BYTES = 64


class SharedMemoryRingBuffer:
    def __init__(self, name: str, capacity: int = 1_048_576, create: bool = False):
        """
        Parameters:
        -----------
        name : str
            Unique OS identifier for the POSIX shared memory block.
        capacity : int
            Number of ticks the buffer can hold. MUST be a power of 2.
            Default: 2^20 = 1,048,576 ticks (~40 MB).
        create : bool
            True if this process allocates the memory (Producer).
            False if attaching to existing memory (Consumer/Worker).
        """
        # 1. Enforce power-of-2 capacity for single-cycle bitwise modulo
        if (capacity & (capacity - 1)) != 0 or capacity <= 0:
            raise ValueError(f"Capacity must be a power of 2, got {capacity}")

        self.name = name
        self.capacity = capacity
        self.mask = capacity - 1
        self.create = create

        # Total memory = 64-byte header + (N * 40 bytes)
        self.data_size_bytes = self.capacity * TICK_ITEMSIZE
        self.total_size_bytes = HEADER_SIZE_BYTES + self.data_size_bytes

        # 2. Acquire or Attach OS Shared Memory
        if self.create:
            # Unlink any stale previous block with the same name if it exists
            try:
                stale_shm = shared_memory.SharedMemory(name=self.name)
                stale_shm.close()
                stale_shm.unlink()
            except FileNotFoundError:
                pass

            self.shm = shared_memory.SharedMemory(
                name=self.name, 
                create=True, 
                size=self.total_size_bytes
            )
            # Initialize header to zeros
            self.shm.buf[:HEADER_SIZE_BYTES] = b'\x00' * HEADER_SIZE_BYTES
        else:
            self.shm = shared_memory.SharedMemory(name=self.name, create=False)

        # 3. Map Zero-Copy Views directly over shared RAM
        # Header view: 2x int64 fields [write_head, is_active]
        self.header = np.ndarray(shape=(2,), dtype=np.int64, buffer=self.shm.buf, offset=0)
        
        # Ring buffer data view: (capacity,) ticks starting at byte 64
        self.buffer = np.ndarray(
            shape=(self.capacity,),
            dtype=TICK_DTYPE,
            buffer=self.shm.buf,
            offset=HEADER_SIZE_BYTES
        )

    def write_tick(self, tick: np.ndarray) -> None:
        """
        Producer method: Writes a single tick into the ring buffer with zero copy.
        Advances write_head monotonically.
        """
        current_seq = self.header[0]
        # Bitwise wrap: equivalent to current_seq % self.capacity, but 1 CPU cycle
        slot = current_seq & self.mask
        
        # Direct write into physical shared memory
        self.buffer[slot] = tick
        
        # Advance the monotonic sequence counter
        self.header[0] = current_seq + 1

    def write_batch(self, batch: np.ndarray) -> None:
        """
        Producer method: Writes a contiguous array of ticks into the ring buffer.
        Handles wrapping around the end of the buffer automatically.
        """
        n = len(batch)
        if n > self.capacity:
            raise ValueError("Batch size exceeds ring buffer capacity")

        current_seq = self.header[0]
        slot = current_seq & self.mask

        if slot + n <= self.capacity:
            # Contiguous chunk: single memcpy
            self.buffer[slot : slot + n] = batch
        else:
            # Chunk wraps around boundary: two split memcpy operations
            first_chunk = self.capacity - slot
            second_chunk = n - first_chunk
            self.buffer[slot : self.capacity] = batch[:first_chunk]
            self.buffer[:second_chunk] = batch[first_chunk:]

        self.header[0] = current_seq + n

    @property
    def write_head(self) -> int:
        """Returns the total number of ticks written so far."""
        return self.header[0]

    def read_latest(self) -> np.ndarray:
        """
        Consumer method: Returns the most recently written tick.
        """
        current_seq = self.header[0]
        if current_seq == 0:
            raise ValueError("Buffer is empty")
        slot = (current_seq - 1) & self.mask
        return self.buffer[slot]

    def read_last_n(self, n: int) -> np.ndarray:
        """
        Consumer method: Returns the last N ticks as a contiguous array view.
        """
        if n > self.capacity:
            raise ValueError(f"Requested {n} ticks, exceeds capacity {self.capacity}")

        current_seq = self.header[0]
        if current_seq < n:
            n = current_seq

        start_seq = current_seq - n
        start_slot = start_seq & self.mask
        end_slot = (current_seq - 1) & self.mask

        if start_slot <= end_slot:
            return self.buffer[start_slot : end_slot + 1]
        else:
            # Handles boundary wrap by concatenating the two contiguous views
            return np.concatenate((self.buffer[start_slot:], self.buffer[: end_slot + 1]))

    def close(self) -> None:
        """Close this process's view of the shared memory."""
        self.shm.close()

    def unlink(self) -> None:
        """OS Kernel cleanup: Permanently frees the physical shared memory segment."""
        if self.create:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass