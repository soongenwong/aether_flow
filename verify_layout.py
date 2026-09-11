# verify_layout.py
import numpy as np
from protocol import TICK_DTYPE

print(f"Item size: {TICK_DTYPE.itemsize} bytes (Must be exactly 40)")
print(f"Alignment: {TICK_DTYPE.alignment}")
print(f"Offsets:   {[TICK_DTYPE.fields[name][1] for name in TICK_DTYPE.names]}")

# Test zero-copy binary reconstruction
raw_bytes = np.zeros(1, dtype=TICK_DTYPE).tobytes()
assert len(raw_bytes) == 40, f"Expected 40 bytes, got {len(raw_bytes)}"

# Wrap raw memory directly into an ndarray view (mental prep for Day 2)
reconstructed = np.frombuffer(raw_bytes, dtype=TICK_DTYPE)
assert reconstructed['timestamp_ns'][0] == 0
print("Memory layout verification: PASSED")