from __future__ import annotations

import numpy as np

from .algorithms import demosaic_ari, demosaic_ari2, demosaic_mlri, demosaic_mlri2, demosaic_ri
from .bayer import mosaic_bayer, parse_code
from .matlab_compat import clip


_ALGORITHMS = {
    "RI": demosaic_ri,
    "MLRI": demosaic_mlri,
    "MLRI2": demosaic_mlri2,
    "ARI": demosaic_ari,
    "ARI2": demosaic_ari2,
}


def demosaic(img_bgr: np.ndarray, code: str) -> np.ndarray:
    """Demosaic a full BGR image using a custom OpenCV-style RI-family code.

    The input is treated as a reference full-color BGR image: a Bayer mosaic is
    generated internally using the pattern encoded in ``code``, then the chosen
    algorithm reconstructs a BGR float64 image.
    """
    parsed = parse_code(code)
    img_bgr = np.asarray(img_bgr)
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must have shape (height, width, 3)")

    rgb = img_bgr[:, :, ::-1].astype(np.float64, copy=False)
    mosaic, mask = mosaic_bayer(rgb, parsed.pattern)
    algorithm = _ALGORITHMS[parsed.algorithm]
    demosaicked_rgb = clip(algorithm(mosaic, mask, parsed.pattern), 0, 255)
    return demosaicked_rgb[:, :, ::-1].astype(np.float64, copy=False)
