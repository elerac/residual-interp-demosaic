from __future__ import annotations

import numpy as np

from .algorithms import demosaic_ari, demosaic_ari2, demosaic_mlri, demosaic_mlri2, demosaic_ri
from .bayer import parse_code
from .matlab_compat import clip


_ALGORITHMS = {
    "RI": demosaic_ri,
    "MLRI": demosaic_mlri,
    "MLRI2": demosaic_mlri2,
    "ARI": demosaic_ari,
    "ARI2": demosaic_ari2,
}

_RGB_CHANNELS = {"r": 0, "g": 1, "b": 2}


def demosaic(cfa: np.ndarray, code: str) -> np.ndarray:
    """Demosaic a single-channel Bayer/CFA frame.

    The pattern and RI-family algorithm are selected by the OpenCV-style code.
    The returned image is a BGR ``float64`` array clipped to the ``0..255``
    range.
    """
    parsed = parse_code(code)
    cfa = np.asarray(cfa)
    if cfa.ndim == 3 and cfa.shape[2] == 3:
        raise ValueError(
            "cfa must be a single-channel Bayer frame with shape (height, width); "
            "3-channel BGR inputs are no longer supported. Use "
            "demosaic.utils.mosaicing_cfa_bayer(image_bgr, pattern) first."
        )
    if cfa.ndim != 2:
        raise ValueError("cfa must be a single-channel Bayer frame with shape (height, width)")

    mosaic, mask = _mosaic_from_cfa(cfa, parsed.pattern)
    algorithm = _ALGORITHMS[parsed.algorithm]
    demosaicked_rgb = clip(algorithm(mosaic, mask, parsed.pattern), 0, 255)
    return demosaicked_rgb[:, :, ::-1].astype(np.float64, copy=False)


def _mosaic_from_cfa(cfa: np.ndarray, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    cfa_float = np.asarray(cfa, dtype=np.float64)
    mosaic = np.zeros((*cfa_float.shape, 3), dtype=np.float64)
    mask = np.zeros_like(mosaic)
    phases = (
        (slice(0, None, 2), slice(0, None, 2), pattern[0]),
        (slice(0, None, 2), slice(1, None, 2), pattern[1]),
        (slice(1, None, 2), slice(0, None, 2), pattern[2]),
        (slice(1, None, 2), slice(1, None, 2), pattern[3]),
    )
    for rows, cols, channel_name in phases:
        channel = _RGB_CHANNELS[channel_name]
        mosaic[rows, cols, channel] = cfa_float[rows, cols]
        mask[rows, cols, channel] = 1.0
    return mosaic, mask
