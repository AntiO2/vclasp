"""Plan coalesced object-store ranges with VClasp."""

import vclasp


records = [
    (10, 4096, 800),
    (11, 6000, 900),
]

plans = vclasp.plan_byte_ranges(
    records,
    merge_threshold_bytes=16 * 1024,
    max_range_bytes=None,
)

for offset, length, members in plans:
    print(f"range offset={offset} length={length} records={members}")
