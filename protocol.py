# protocol.py
import numpy as np

# Total size: 8 * 5 = 40 bytes per tick
# Flat 64-bit alignment ensures zero padding and direct C-struct compatibility
TICK_DTYPE = np.dtype([
    ('timestamp_ns', np.int64),   # 8 bytes: Unix epoch nanoseconds
    ('bid_price',    np.float64), # 8 bytes: Best bid
    ('ask_price',    np.float64), # 8 bytes: Best ask
    ('bid_size',     np.float64), # 8 bytes: Volume at best bid
    ('ask_size',     np.float64), # 8 bytes: Volume at best ask
], align=True)

TICK_ITEMSIZE = TICK_DTYPE.itemsize
EXPECTED_OFFSETS = {
    'timestamp_ns': 0,
    'bid_price'   : 8,
    'ask_price'   : 16,
    'bid_size'    : 24,
    'ask_size'    : 32
}

assert TICK_ITEMSIZE == 40, f"Protocol integrity broken: itemsize is {TICK_ITEMSIZE}, expected 40"