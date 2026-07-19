"""Inspect a VClasp format-v1 chunk."""

from pathlib import Path
import argparse

import vclasp


parser = argparse.ArgumentParser()
parser.add_argument("chunk", type=Path)
args = parser.parse_args()

chunk = vclasp.VClaspChunk(str(args.chunk))
print("format version:", chunk.format_version())
print("records:", chunk.record_count())
print("codec:", chunk.codec_info())
