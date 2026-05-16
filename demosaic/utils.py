from __future__ import annotations

import numpy as np

from .bayer import PATTERNS


_BGR_CHANNELS = {"r": 2, "g": 1, "b": 0}


def mosaicing_cfa_bayer(image_bgr: np.ndarray, pattern: str) -> np.ndarray:
    """Create a single-channel Bayer/CFA frame from an OpenCV BGR image."""
    image_bgr = np.asarray(image_bgr)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3)")

    pattern = pattern.lower()
    if pattern not in PATTERNS:
        raise ValueError(f"unsupported Bayer pattern: {pattern!r}")

    cfa = np.empty(image_bgr.shape[:2], dtype=image_bgr.dtype)
    phases = (
        (slice(0, None, 2), slice(0, None, 2), pattern[0]),
        (slice(0, None, 2), slice(1, None, 2), pattern[1]),
        (slice(1, None, 2), slice(0, None, 2), pattern[2]),
        (slice(1, None, 2), slice(1, None, 2), pattern[3]),
    )
    for rows, cols, channel_name in phases:
        cfa[rows, cols] = image_bgr[rows, cols, _BGR_CHANNELS[channel_name]]
    return cfa
