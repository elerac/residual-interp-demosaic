from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def clip(x: np.ndarray, lo: float = 0.0, hi: float = 255.0) -> np.ndarray:
    return np.clip(x, lo, hi)


def imfilter(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """MATLAB imfilter(src, kernel, 'replicate') with correlation semantics."""
    src = np.asarray(src, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim == 1:
        kernel = kernel.reshape(1, -1)
    if kernel.ndim != 2:
        raise ValueError("kernel must be 1D or 2D")
    if src.ndim == 3:
        return np.stack([imfilter(src[:, :, channel], kernel) for channel in range(src.shape[2])], axis=2)
    return ndimage.correlate(src, kernel, mode="nearest")


def boxfilter(src: np.ndarray, h: int, v: int) -> np.ndarray:
    """Windowed sum equivalent to the MATLAB boxfilter helper.

    h is the horizontal radius and v is the vertical radius. Outside-image
    samples are omitted, which is equivalent to zero padding for sums.
    """
    src = np.asarray(src, dtype=np.float64)
    if h == 0 and v == 0:
        return np.zeros_like(src)
    if src.ndim == 3:
        return np.stack([boxfilter(src[:, :, channel], h, v) for channel in range(src.shape[2])], axis=2)
    kernel = np.ones((2 * v + 1, 2 * h + 1), dtype=np.float64)
    return ndimage.correlate(src, kernel, mode="constant", cval=0.0)


def gaussian_kernel(size: tuple[int, int], sigma: float) -> np.ndarray:
    rows, cols = size
    y = np.arange(rows, dtype=np.float64) - (rows - 1) / 2.0
    x = np.arange(cols, dtype=np.float64) - (cols - 1) / 2.0
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    total = np.sum(kernel)
    if total != 0:
        kernel = kernel / total
    return kernel


def filter2_valid(kernel: np.ndarray, image: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    return signal.correlate2d(image, kernel, mode="valid")


def diagonal_window(h: int, v: int) -> np.ndarray:
    r = h + v
    w = 2 * r + 1
    f = np.ones((w, w), dtype=np.float64)
    for i in range(1, v + 1):
        for t in range(1, 2 * i):
            f[t - 1, 2 * i - t - 1] = 0.0
            f[w - t, w - 2 * i + t] = 0.0
    for i in range(1, h + 1):
        for t in range(1, 2 * i):
            f[t - 1, w - 2 * i + t] = 0.0
            f[w - t, 2 * i - t - 1] = 0.0
    f2 = np.zeros((w, w), dtype=np.float64)
    f2[0::2, 0::2] = 1.0
    f2[1::2, 1::2] = 1.0
    return f * f2


def cubic_s(x: float) -> float:
    ax = abs(x)
    if ax <= 1:
        return 1.5 * ax**3 - 2.5 * ax**2 + 1
    if ax <= 2:
        return -0.5 * ax**3 + 2.5 * ax**2 - 4 * ax + 2
    return 0.0
