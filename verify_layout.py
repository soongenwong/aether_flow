import sys
import numpy as np
from protocol import TICK_DTYPE, TICK_ITEMSIZE, EXPECTED_OFFSETS

def test_byte_offsets():
    print("[1/4] Verifying Byte Offsets and Alignment...")
    for field_name, expected_offset in EXPECTED_OFFSETS.items():
        field_dtype, actual_offset = TICK_DTYPE.fields[field_name][:2]
        
        assert actual_offset == expected_offset, (
            f"Offset mismatch for {field_name}: expected {expected_offset}, got {actual_offset}"
        )
        print(f"  - Field '{field_name:<12}' -> Offset {actual_offset:>2} bytes [OK]")

    assert TICK_DTYPE.alignment == 8, f"Alignment is {TICK_DTYPE.alignment}, expected 8"
    print(f"  Total Itemsize: {TICK_ITEMSIZE} bytes (Alignment: {TICK_DTYPE.alignment}-byte aligned)\n")

def test_zero_copy_view_casting():
    print("[2/4] Testing Zero-Copy In-Memory View Mutation...")

    original_tick = np.zeros(1, dtype = TICK_DTYPE)
    original_tick['timestamp_ns'] = 1_700_000_000_000_000_000
    original_tick['bid_price'] = 100.50
    original_tick['ask_price'] = 100.55
    original_tick['bid_size'] = 250.00
    original_tick['ask_size'] = 300.00

    raw_buffer = original_tick.data
    view_tick = np.ndarray(shape = (1,), dtype = TICK_DTYPE, buffer = raw_buffer)
    assert view_tick['bid_price'][0] == 100.50

    view_tick['bid_price'][0] = 105.75
    assert original_tick['bid_price'][0] == 105.75
    print ("    Zero-copy view mutation verified: original and view share identical memory.\n")

def test_serialisation_round_trip():
    print("[3/4] Testing raw binary serialisation...")
    n_ticks = 1000
    ticks = np.zeros(n_ticks, dtype = TICK_DTYPE)

    byte_payload = ticks.tobytes()
    expected_bytes = n_ticks * 40
    assert len(byte_payload) == expected_bytes, f"Expected {expected_bytes} bytes, got{len(byte_payload)}"

    reconstructed = np.frombuffer(byte_payload, dtype = TICK_DTYPE)
    assert len(reconstructed) == n_ticks
    print(f"    Seriaslied and deserialised {n_ticks} ticks ({expected_bytes:,} raw_bytes) cleanly.\n")

def test_cache_line_footprint():
    print("[4/4] Hardware Cache Line Footprint Analysis...")
    cache_line_size = 64
    ticks_per_line = cache_line_size / TICK_ITEMSIZE
    print(f"    CPU L1/L2/L3 Cache Line Size: {cache_line_size} bytes")
    print(f"    Ticks per Cache Line: {ticks_per_line:.2f}")
    print(f"    A 1,000,000-tick circular buffer requires: {(1_000_000 * TICK_ITEMSIZE) / (1024 * 1024): .2f} MB RAM\n") 

if __name__ == "__main__":
    print("==================================================")
    print("      AETHERFLOW PROTOCOL VERIFICATION SUITE       ")
    print("==================================================\n")
    try:
        test_byte_offsets()
        test_zero_copy_view_casting()
        test_serialisation_round_trip()
        test_cache_line_footprint()
        print(">>> ALL ARCHITECTURAL CHECKS PASSED SUCCESSFULLY. <<<")
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)

