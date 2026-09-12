## Day 1: 
Engineered a vectorized Poisson-GBM synthetic tick feed delivering 7.6M+ ticks/sec (batch) and 50k ticks/sec (paced streaming)

### python verify_layout.py

[1/4] Verifying Byte Offsets and Alignment...
  - Field 'timestamp_ns' -> Offset  0 bytes [OK]
  - Field 'bid_price   ' -> Offset  8 bytes [OK]
  - Field 'ask_price   ' -> Offset 16 bytes [OK]
  - Field 'bid_size    ' -> Offset 24 bytes [OK]
  - Field 'ask_size    ' -> Offset 32 bytes [OK]
  Total Itemsize: 40 bytes (Alignment: 8-byte aligned)

[2/4] Testing Zero-Copy In-Memory View Mutation...
    Zero-copy view mutation verified: original and view share identical memory.

[3/4] Testing raw binary serialisation...
    Seriaslied and deserialised 1000 ticks (40,000 raw_bytes) cleanly.

[4/4] Hardware Cache Line Footprint Analysis...
    CPU L1/L2/L3 Cache Line Size: 64 bytes
    Ticks per Cache Line: 1.60
    A 1,000,000-tick circular buffer requires:  38.15 MB RAM

>>> ALL ARCHITECTURAL CHECKS PASSED SUCCESSFULLY. <<<

### python market_generator.py

stream: 250,001 ticks in 5.000016s (50,000 ticks/sec)
batch:  1,000,000 ticks in 0.130050s (7,689,335 ticks/sec)
dtype:   True
spread:  ask_price > bid_price verified



## Day 2:

