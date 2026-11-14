"""Deterministic per-segment shift of the success boundary, in source frames."""

import hashlib
import json
import math

import numpy as np


def boundary_offset(segment_id, fps, radius_seconds, seed):
    if (
        not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(radius_seconds)
        or radius_seconds < 0
    ):
        raise ValueError("FPS must be positive and jitter radius nonnegative")
    maximum = math.floor(fps * radius_seconds + 1e-9)
    payload = json.dumps([int(seed), str(segment_id)], separators=(",", ":")).encode()
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    )
    return int(rng.integers(-maximum, maximum + 1))
