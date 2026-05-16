"""Adaptive residual interpolation demosaicing.

This module translates the MATLAB ARI and ARI2 implementations under
``matlab/algorithms/ARI`` and ``matlab/algorithms/ARI2``.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

try:
    from demosaic.matlab_compat import boxfilter as _shared_boxfilter
    from demosaic.matlab_compat import imfilter as _shared_imfilter
except Exception:  # pragma: no cover - used only while shared primitives are absent.
    _shared_boxfilter = None
    _shared_imfilter = None

try:
    from demosaic.bayer import mask_gr_gb as _shared_mask_gr_gb
except Exception:  # pragma: no cover - used only while shared primitives are absent.
    _shared_mask_gr_gb = None

_BOXFILTER_MANY_MAX_ELEMENTS = 4_000_000
_BAYER_PHASES = {
    "rggb": {"r": (0, 0), "gr": (0, 1), "gb": (1, 0), "b": (1, 1)},
    "grbg": {"gr": (0, 0), "r": (0, 1), "b": (1, 0), "gb": (1, 1)},
    "gbrg": {"gb": (0, 0), "b": (0, 1), "r": (1, 0), "gr": (1, 1)},
    "bggr": {"b": (0, 0), "gb": (0, 1), "gr": (1, 0), "r": (1, 1)},
}


def _readonly_kernel(values: np.ndarray | list) -> np.ndarray:
    kernel = np.asarray(values, dtype=np.float64)
    kernel.setflags(write=False)
    return kernel


_ARI_HALF_H = _readonly_kernel([[0.5, 0, 0.5]])
_ARI_HALF_V = _readonly_kernel(_ARI_HALF_H.T.copy())
_ARI_DETAIL_H = _readonly_kernel([[-1, 0, 2, 0, -1]])
_ARI_DETAIL_V = _readonly_kernel(_ARI_DETAIL_H.T.copy())
_ARI_GREEN_RESIDUAL_H = _readonly_kernel([[0.5, 1, 0.5]])
_ARI_GREEN_RESIDUAL_V = _readonly_kernel(_ARI_GREEN_RESIDUAL_H.T.copy())
_ARI_CRI_H = _readonly_kernel([[-1, 0, 1]])
_ARI_CRI_V = _readonly_kernel(_ARI_CRI_H.T.copy())
_ARI_LAPLACIAN = _readonly_kernel(
    [
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [-1, 0, 4, 0, -1],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
    ]
)
_ARI_DIAG_HALF_1 = _readonly_kernel([[0.5, 0, 0], [0, 0, 0], [0, 0, 0.5]])
_ARI_DIAG_HALF_2 = _readonly_kernel([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]])
_ARI_DIAG_DETAIL_1 = _readonly_kernel(
    [[-1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, -1]]
)
_ARI_DIAG_DETAIL_2 = _readonly_kernel(
    [[0, 0, 0, 0, -1], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0]]
)
_ARI_DIAG_CRI_1 = _readonly_kernel([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
_ARI_DIAG_CRI_2 = _readonly_kernel([[0, 0, -1], [0, 0, 0], [1, 0, 0]])


def demosaic_ari(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    """Demosaic a Bayer image with ARI, returning RGB ``float64``."""
    mosaic, mask = _as_float_inputs(mosaic, mask)
    eps = 1e-10
    h = 5
    v = 5

    green = _green_interpolation(mosaic, mask, pattern, eps)
    red = _red_interpolation(green, mosaic, mask, pattern, h, v, eps)
    blue = _blue_interpolation(green, mosaic, mask, pattern, h, v, eps)

    rgb_dem = np.empty_like(mosaic, dtype=np.float64)
    rgb_dem[:, :, 0] = red
    rgb_dem[:, :, 1] = green
    rgb_dem[:, :, 2] = blue
    return rgb_dem


def demosaic_ari2(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    """Demosaic a Bayer image with ARI2, returning RGB ``float64``."""
    mosaic, mask = _as_float_inputs(mosaic, mask)
    eps = 1e-10

    green = _green_interpolation(mosaic, mask, pattern, eps)
    red, blue = _red_blue_interpolation_first(green, mosaic, mask, pattern, eps)
    red, blue = _red_blue_interpolation_second(green, red, blue, mask, pattern, eps)

    rgb_dem = np.empty_like(mosaic, dtype=np.float64)
    rgb_dem[:, :, 0] = red
    rgb_dem[:, :, 1] = green
    rgb_dem[:, :, 2] = blue
    return rgb_dem


def _as_float_inputs(mosaic: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mosaic = np.asarray(mosaic, dtype=np.float64)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        mask = mask != 0
    if mosaic.ndim != 3 or mask.ndim != 3 or mosaic.shape != mask.shape or mosaic.shape[2] != 3:
        raise ValueError("mosaic and mask must both have shape (height, width, 3)")
    return mosaic, mask


def _imfilter(src: np.ndarray, kernel: np.ndarray | list, boundary: str = "replicate") -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim == 1:
        kernel = kernel.reshape(1, -1)
    if boundary != "replicate":
        raise ValueError("only replicate boundary is supported")
    if _needs_exact_scipy_imfilter(kernel):
        src = np.asarray(src, dtype=np.float64)
        if src.ndim == 3:
            from scipy import ndimage

            return ndimage.correlate(src, kernel[:, :, None], mode="nearest")
        if _shared_imfilter is not None:
            return np.asarray(_shared_imfilter(src, kernel), dtype=np.float64)

        from scipy import ndimage

        return ndimage.correlate(src, kernel, mode="nearest")

    src = np.asarray(src, dtype=np.float64)
    if src.ndim == 3:
        return cv2.filter2D(src, cv2.CV_64F, kernel, borderType=cv2.BORDER_REPLICATE)
    return cv2.filter2D(src, cv2.CV_64F, kernel, borderType=cv2.BORDER_REPLICATE)


def _imfilter_many(srcs: tuple[np.ndarray, ...], kernel: np.ndarray | list, boundary: str = "replicate") -> tuple[np.ndarray, ...]:
    return tuple(_imfilter(src, kernel, boundary) for src in srcs)


def _needs_exact_scipy_imfilter(kernel: np.ndarray) -> bool:
    # OpenCV accumulation can flip ARI2 adaptive choices for diagonal support windows.
    return (
        kernel.ndim == 2
        and kernel.shape[0] == kernel.shape[1]
        and kernel.shape[0] >= 5
        and np.any(kernel == 0.0)
        and np.any(kernel == 1.0)
        and np.all((kernel == 0.0) | (kernel == 1.0))
    )


def _boxfilter(src: np.ndarray, h: int, v: int) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    if h == 0 and v == 0:
        return np.zeros_like(src)
    hei, wid = src.shape
    integral = np.empty((hei + 1, wid + 1), dtype=np.float64)
    integral[0, :] = 0.0
    integral[:, 0] = 0.0
    integral[1:, 1:] = src
    np.cumsum(integral, axis=0, out=integral)
    np.cumsum(integral, axis=1, out=integral)
    r0, r1, c0, c1 = _boxfilter_indices((hei, wid), h, v)
    out = integral[r1[:, None], c1[None, :]] - integral[r0[:, None], c1[None, :]]
    out -= integral[r1[:, None], c0[None, :]]
    out += integral[r0[:, None], c0[None, :]]
    return out


def _boxfilter_many(srcs: tuple[np.ndarray, ...], h: int, v: int) -> tuple[np.ndarray, ...]:
    if len(srcs) < 5:
        return tuple(_boxfilter(src, h, v) for src in srcs)
    first = np.asarray(srcs[0], dtype=np.float64)
    if first.size * len(srcs) > _BOXFILTER_MANY_MAX_ELEMENTS:
        return tuple(_boxfilter(src, h, v) for src in srcs)
    if h == 0 and v == 0:
        return tuple(np.zeros_like(np.asarray(src, dtype=np.float64)) for src in srcs)

    hei, wid = first.shape
    integral = np.empty((hei + 1, wid + 1, len(srcs)), dtype=np.float64)
    integral[0, :, :] = 0.0
    integral[:, 0, :] = 0.0
    integral[1:, 1:, 0] = first
    for index, src in enumerate(srcs[1:], start=1):
        integral[1:, 1:, index] = np.asarray(src, dtype=np.float64)
    np.cumsum(integral, axis=0, out=integral)
    np.cumsum(integral, axis=1, out=integral)
    r0, r1, c0, c1 = _boxfilter_indices((hei, wid), h, v)
    out = integral[r1[:, None], c1[None, :], :] - integral[r0[:, None], c1[None, :], :]
    out -= integral[r1[:, None], c0[None, :], :]
    out += integral[r0[:, None], c0[None, :], :]
    return tuple(out[:, :, index] for index in range(len(srcs)))


@lru_cache(maxsize=None)
def _boxfilter_indices(shape: tuple[int, int], h: int, v: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hei, wid = shape
    rows = np.arange(hei)
    cols = np.arange(wid)
    r0 = np.maximum(rows - v, 0)
    r1 = np.minimum(rows + v + 1, hei)
    c0 = np.maximum(cols - h, 0)
    c1 = np.minimum(cols + h + 1, wid)
    return r0, r1, c0, c1


@lru_cache(maxsize=None)
def _mask_gr_gb(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, np.ndarray]:
    pattern = pattern.lower()
    if _shared_mask_gr_gb is not None:
        mask_gr, mask_gb = _shared_mask_gr_gb(shape, pattern)
        mask_gr = np.asarray(mask_gr, dtype=bool)
        mask_gb = np.asarray(mask_gb, dtype=bool)
        mask_gr.setflags(write=False)
        mask_gb.setflags(write=False)
        return mask_gr, mask_gb

    mask_gr = np.zeros(shape, dtype=bool)
    mask_gb = np.zeros(shape, dtype=bool)
    if pattern == "grbg":
        mask_gr[0::2, 0::2] = True
        mask_gb[1::2, 1::2] = True
    elif pattern == "rggb":
        mask_gr[0::2, 1::2] = True
        mask_gb[1::2, 0::2] = True
    elif pattern == "gbrg":
        mask_gb[0::2, 0::2] = True
        mask_gr[1::2, 1::2] = True
    elif pattern == "bggr":
        mask_gb[0::2, 1::2] = True
        mask_gr[1::2, 0::2] = True
    else:
        raise ValueError(f"unsupported Bayer pattern: {pattern!r}")
    mask_gr.setflags(write=False)
    mask_gb.setflags(write=False)
    return mask_gr, mask_gb


@lru_cache(maxsize=None)
def _gaussian_kernel(size: tuple[int, int] = (5, 5), sigma: float = 2.0) -> np.ndarray:
    rows, cols = size
    y = np.arange(rows, dtype=np.float64) - (rows - 1) / 2
    x = np.arange(cols, dtype=np.float64) - (cols - 1) / 2
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel.setflags(write=False)
    return kernel


def _clip(image: np.ndarray, low: float = 0.0, high: float = 255.0) -> np.ndarray:
    return np.clip(image, low, high)


def _s(x: float) -> float:
    a = -0.5
    ax = abs(x)
    if 2 > x > 1:
        return a * ax**3 - 5 * a * ax**2 + 8 * a * ax - 4 * a
    if a >= 2:
        return 0.0
    return (a + 2) * ax**3 - (a + 3) * ax**2 + 1


def _boxfilter_count(M: np.ndarray, h: int, v: int) -> np.ndarray:
    # Counts are sums of binary masks, so OpenCV's separable accumulation is
    # exact here and avoids the slower ARI integral path used for image data.
    N = _shared_boxfilter(M, h, v) if _shared_boxfilter is not None else _boxfilter(M, h, v)
    N[N == 0] = 1
    return N


@lru_cache(maxsize=None)
def _boxfilter_ones_count(shape: tuple[int, int], h: int, v: int) -> np.ndarray:
    r0, r1, c0, c1 = _boxfilter_indices(shape, h, v)
    N = (r1 - r0)[:, None].astype(np.float64) * (c1 - c0)[None, :].astype(np.float64)
    N.setflags(write=False)
    return N


@lru_cache(maxsize=None)
def _bayer_count(shape: tuple[int, int], pattern: str, key: str, h: int, v: int) -> np.ndarray:
    row_phase, col_phase = _BAYER_PHASES[pattern.lower()][key]
    N = _bayer_phase_raw_count(shape, row_phase, col_phase, h, v).copy()
    N[N == 0] = 1
    N.setflags(write=False)
    return N


@lru_cache(maxsize=None)
def _bayer_pair_count(shape: tuple[int, int], pattern: str, key1: str, key2: str, h: int, v: int) -> np.ndarray:
    phases = _BAYER_PHASES[pattern.lower()]
    row_phase1, col_phase1 = phases[key1]
    row_phase2, col_phase2 = phases[key2]
    N = (
        _bayer_phase_raw_count(shape, row_phase1, col_phase1, h, v)
        + _bayer_phase_raw_count(shape, row_phase2, col_phase2, h, v)
    )
    N = N.copy()
    N[N == 0] = 1
    N.setflags(write=False)
    return N


@lru_cache(maxsize=None)
def _bayer_phase_raw_count(shape: tuple[int, int], row_phase: int, col_phase: int, h: int, v: int) -> np.ndarray:
    rows = _parity_window_counts(shape[0], v, row_phase)
    cols = _parity_window_counts(shape[1], h, col_phase)
    N = rows[:, None].astype(np.float64) * cols[None, :].astype(np.float64)
    N.setflags(write=False)
    return N


def _parity_window_counts(length: int, radius: int, phase: int) -> np.ndarray:
    positions = np.arange(length, dtype=np.int64)
    lo = np.maximum(positions - radius, 0)
    hi = np.minimum(positions + radius, length - 1)
    return _parity_count_through(hi, phase) - _parity_count_through(lo - 1, phase)


def _parity_count_through(index: np.ndarray, phase: int) -> np.ndarray:
    if phase == 0:
        counts = index // 2 + 1
    else:
        counts = (index + 1) // 2
    return np.where(index >= 0, counts, 0)


def _imfilter_count(M: np.ndarray, F: np.ndarray) -> np.ndarray:
    N = _imfilter(M, F, "replicate")
    N[N == 0] = 1
    return N


@lru_cache(maxsize=None)
def _diagonal_bayer_count(shape: tuple[int, int], pattern: str, keys: tuple[str, ...], h: int, v: int) -> np.ndarray:
    M = np.zeros(shape, dtype=np.float64)
    phases = _BAYER_PHASES[pattern.lower()]
    for key in keys:
        row_phase, col_phase = phases[key]
        M[row_phase::2, col_phase::2] = 1.0
    N = _imfilter_count(M, _diagonal_window(h, v))
    N.setflags(write=False)
    return N


def _guidedfilter(
    I: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray | None = None,
) -> np.ndarray:
    if N is None:
        N = _boxfilter_count(M, h, v)

    IM = I * M
    pM = p * M
    sum_I, sum_p, sum_Ip, sum_II, sum_pp = _boxfilter_many(
        (IM, pM, I * pM, I * IM, p * pM), h, v
    )
    mean_I = sum_I / N
    mean_p = sum_p / N
    mean_Ip = sum_Ip / N

    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = sum_II / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    dif = (
        sum_II * a * a
        + b * b * N
        + sum_pp
        + 2 * a * b * sum_I
        - 2 * b * sum_p
        - 2 * a * sum_Ip
    )
    dif = np.sqrt(np.maximum(dif / N, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif
    wdif = _boxfilter(dif, h, v)
    mean_a = _boxfilter(a * dif, h, v) / (wdif + 1e-4)
    mean_b = _boxfilter(b * dif, h, v) / (wdif + 1e-4)

    return mean_a * I + mean_b


def _guidedfilter_same_guide(
    I: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    M: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    IM = I * M
    p1M = p1 * M
    p2M = p2 * M
    sum_I, sum_II, sum_p1, sum_Ip1, sum_pp1, sum_p2, sum_Ip2, sum_pp2 = _boxfilter_many(
        (IM, I * IM, p1M, I * p1M, p1 * p1M, p2M, I * p2M, p2 * p2M), h, v
    )
    mean_I = sum_I / N
    mean_II = sum_II / N
    var_I = mean_II - mean_I * mean_I
    denom = var_I + eps

    mean_p1 = sum_p1 / N
    mean_Ip1 = sum_Ip1 / N
    cov_Ip1 = mean_Ip1 - mean_I * mean_p1
    a1 = cov_Ip1 / denom
    b1 = mean_p1 - a1 * mean_I

    mean_p2 = sum_p2 / N
    mean_Ip2 = sum_Ip2 / N
    cov_Ip2 = mean_Ip2 - mean_I * mean_p2
    a2 = cov_Ip2 / denom
    b2 = mean_p2 - a2 * mean_I

    dif1 = sum_II * a1 * a1 + b1 * b1 * N + sum_pp1 + 2 * a1 * b1 * sum_I - 2 * b1 * sum_p1 - 2 * a1 * sum_Ip1
    dif1 = np.sqrt(np.maximum(dif1 / N, 0))
    dif1[dif1 < 1e-3] = 1e-3
    dif1 = 1 / dif1

    dif2 = sum_II * a2 * a2 + b2 * b2 * N + sum_pp2 + 2 * a2 * b2 * sum_I - 2 * b2 * sum_p2 - 2 * a2 * sum_Ip2
    dif2 = np.sqrt(np.maximum(dif2 / N, 0))
    dif2[dif2 < 1e-3] = 1e-3
    dif2 = 1 / dif2

    wdif1, sum_a1, sum_b1, wdif2, sum_a2, sum_b2 = _boxfilter_many(
        (dif1, a1 * dif1, b1 * dif1, dif2, a2 * dif2, b2 * dif2), h, v
    )
    mean_a1 = sum_a1 / (wdif1 + 1e-4)
    mean_b1 = sum_b1 / (wdif1 + 1e-4)
    mean_a2 = sum_a2 / (wdif2 + 1e-4)
    mean_b2 = sum_b2 / (wdif2 + 1e-4)
    out1 = mean_a1 * I + mean_b1
    out2 = mean_a2 * I + mean_b2
    return out1, out2


def _guidedfilter_pair(
    I: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if N is None:
        N = _boxfilter_count(M, h, v)

    IM = I * M
    pM = p * M
    sum_I, sum_p, sum_Ip, sum_II, sum_pp = _boxfilter_many(
        (IM, pM, I * pM, I * IM, p * pM), h, v
    )

    mean_I = sum_I / N
    mean_p = sum_p / N
    mean_Ip = sum_Ip / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = sum_II / N
    var_I = mean_II - mean_I * mean_I
    a_Ip = cov_Ip / (var_I + eps)
    b_Ip = mean_p - a_Ip * mean_I
    dif_Ip = (
        sum_II * a_Ip * a_Ip
        + b_Ip * b_Ip * N
        + sum_pp
        + 2 * a_Ip * b_Ip * sum_I
        - 2 * b_Ip * sum_p
        - 2 * a_Ip * sum_Ip
    )
    dif_Ip = np.sqrt(np.maximum(dif_Ip / N, 0))
    dif_Ip[dif_Ip < 1e-3] = 1e-3
    dif_Ip = 1 / dif_Ip

    mean_pp = sum_pp / N
    var_p = mean_pp - mean_p * mean_p
    a_pI = cov_Ip / (var_p + eps)
    b_pI = mean_I - a_pI * mean_p
    dif_pI = (
        sum_pp * a_pI * a_pI
        + b_pI * b_pI * N
        + sum_II
        + 2 * a_pI * b_pI * sum_p
        - 2 * b_pI * sum_I
        - 2 * a_pI * sum_Ip
    )
    dif_pI = np.sqrt(np.maximum(dif_pI / N, 0))
    dif_pI[dif_pI < 1e-3] = 1e-3
    dif_pI = 1 / dif_pI

    wdif_Ip, sum_a_Ip, sum_b_Ip, wdif_pI, sum_a_pI, sum_b_pI = _boxfilter_many(
        (dif_Ip, a_Ip * dif_Ip, b_Ip * dif_Ip, dif_pI, a_pI * dif_pI, b_pI * dif_pI), h, v
    )
    mean_a_Ip = sum_a_Ip / (wdif_Ip + 1e-4)
    mean_b_Ip = sum_b_Ip / (wdif_Ip + 1e-4)
    mean_a_pI = sum_a_pI / (wdif_pI + 1e-4)
    mean_b_pI = sum_b_pI / (wdif_pI + 1e-4)
    return mean_a_Ip * I + mean_b_Ip, mean_a_pI * p + mean_b_pI


def _guidedfilter_mlri(
    G: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    I: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray | None = None,
    N3: np.ndarray | None = None,
) -> np.ndarray:
    if N is None:
        N = _boxfilter_count(M, h, v)

    IM = I * M
    pM = p * M
    Gm = G * mask
    Rm = R * mask
    sum_Ip, sum_II, sum_G, sum_R, sum_GG, sum_RR, sum_RG = _boxfilter_many(
        (I * pM, I * IM, Gm, Rm, G * Gm, R * Rm, R * Gm), h, v
    )
    mean_Ip = sum_Ip / N
    mean_II = sum_II / N

    a = mean_Ip / (mean_II + eps)
    if N3 is None:
        N3 = _boxfilter_count(mask, h, v)
    mean_G = sum_G / N3
    mean_R = sum_R / N3
    b = mean_R - a * mean_G

    dif = (
        sum_GG * a * a
        + b * b * N3
        + sum_RR
        + 2 * a * b * sum_G
        - 2 * b * sum_R
        - 2 * a * sum_RG
    )
    dif = np.sqrt(np.maximum(dif / N3, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif
    wdif = _boxfilter(dif, h, v)
    mean_a = _boxfilter(a * dif, h, v) / (wdif + 1e-4)
    mean_b = _boxfilter(b * dif, h, v) / (wdif + 1e-4)

    return mean_a * G + mean_b


def _guidedfilter_mlri_pair(
    G: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    I: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    M_reverse: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray,
    N_reverse: np.ndarray,
    N3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Gm = G * mask
    Rm = R * mask
    sum_G, sum_R, sum_GG, sum_RR, sum_RG = _boxfilter_many((Gm, Rm, G * Gm, R * Rm, R * Gm), h, v)

    IM = I * M
    pM = p * M
    reverse_IM = p * M_reverse
    reverse_pM = I * M_reverse
    sum_Ip, sum_II, sum_Ip_reverse, sum_II_reverse = _boxfilter_many(
        (I * pM, I * IM, p * reverse_pM, p * reverse_IM), h, v
    )

    mean_Ip = sum_Ip / N
    mean_II = sum_II / N
    a = mean_Ip / (mean_II + eps)
    mean_G = sum_G / N3
    mean_R = sum_R / N3
    b = mean_R - a * mean_G
    dif = sum_GG * a * a + b * b * N3 + sum_RR + 2 * a * b * sum_G - 2 * b * sum_R - 2 * a * sum_RG
    dif = np.sqrt(np.maximum(dif / N3, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif

    mean_Ip_reverse = sum_Ip_reverse / N_reverse
    mean_II_reverse = sum_II_reverse / N_reverse
    a_reverse = mean_Ip_reverse / (mean_II_reverse + eps)
    b_reverse = mean_G - a_reverse * mean_R
    dif_reverse = (
        sum_RR * a_reverse * a_reverse
        + b_reverse * b_reverse * N3
        + sum_GG
        + 2 * a_reverse * b_reverse * sum_R
        - 2 * b_reverse * sum_G
        - 2 * a_reverse * sum_RG
    )
    dif_reverse = np.sqrt(np.maximum(dif_reverse / N3, 0))
    dif_reverse[dif_reverse < 1e-3] = 1e-3
    dif_reverse = 1 / dif_reverse

    wdif, sum_a, sum_b, wdif_reverse, sum_a_reverse, sum_b_reverse = _boxfilter_many(
        (dif, a * dif, b * dif, dif_reverse, a_reverse * dif_reverse, b_reverse * dif_reverse), h, v
    )
    mean_a = sum_a / (wdif + 1e-4)
    mean_b = sum_b / (wdif + 1e-4)
    mean_a_reverse = sum_a_reverse / (wdif_reverse + 1e-4)
    mean_b_reverse = sum_b_reverse / (wdif_reverse + 1e-4)
    return mean_a * G + mean_b, mean_a_reverse * R + mean_b_reverse


def _guidedfilter_mlri_same_guide(
    G: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    mask: np.ndarray,
    I: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    M: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray,
    N3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    IM = I * M
    Gm = G * mask
    p1M = p1 * M
    p2M = p2 * M
    R1m = R1 * mask
    R2m = R2 * mask
    sum_II, sum_G, sum_GG, sum_Ip1, sum_R1, sum_RR1, sum_RG1, sum_Ip2, sum_R2, sum_RR2, sum_RG2 = _boxfilter_many(
        (
            I * IM,
            Gm,
            G * Gm,
            I * p1M,
            R1m,
            R1 * R1m,
            R1 * G * mask,
            I * p2M,
            R2m,
            R2 * R2m,
            R2 * G * mask,
        ),
        h,
        v,
    )

    mean_II = sum_II / N
    mean_G = sum_G / N3
    denom = mean_II + eps

    a1 = (sum_Ip1 / N) / denom
    mean_R1 = sum_R1 / N3
    b1 = mean_R1 - a1 * mean_G

    a2 = (sum_Ip2 / N) / denom
    mean_R2 = sum_R2 / N3
    b2 = mean_R2 - a2 * mean_G

    dif1 = sum_GG * a1 * a1 + b1 * b1 * N3 + sum_RR1 + 2 * a1 * b1 * sum_G - 2 * b1 * sum_R1 - 2 * a1 * sum_RG1
    dif1 = np.sqrt(np.maximum(dif1 / N3, 0))
    dif1[dif1 < 1e-3] = 1e-3
    dif1 = 1 / dif1

    dif2 = sum_GG * a2 * a2 + b2 * b2 * N3 + sum_RR2 + 2 * a2 * b2 * sum_G - 2 * b2 * sum_R2 - 2 * a2 * sum_RG2
    dif2 = np.sqrt(np.maximum(dif2 / N3, 0))
    dif2[dif2 < 1e-3] = 1e-3
    dif2 = 1 / dif2

    wdif1, sum_a1, sum_b1, wdif2, sum_a2, sum_b2 = _boxfilter_many(
        (dif1, a1 * dif1, b1 * dif1, dif2, a2 * dif2, b2 * dif2), h, v
    )
    mean_a1 = sum_a1 / (wdif1 + 1e-4)
    mean_b1 = sum_b1 / (wdif1 + 1e-4)
    mean_a2 = sum_a2 / (wdif2 + 1e-4)
    mean_b2 = sum_b2 / (wdif2 + 1e-4)
    out1 = mean_a1 * G + mean_b1
    out2 = mean_a2 * G + mean_b2
    return out1, out2


@lru_cache(maxsize=None)
def _diagonal_window(h: int, v: int) -> np.ndarray:
    r = h + v
    F = np.ones((2 * r + 1, 2 * r + 1), dtype=np.float64)
    w = 2 * r + 1
    for i in range(1, v + 1):
        for t in range(1, 2 * i):
            F[t - 1, 2 * i - t - 1] = 0
            F[w - t, w - 2 * i + t] = 0
    for i in range(1, h + 1):
        for t in range(1, 2 * i):
            F[t - 1, w - 2 * i + t] = 0
            F[w - t, 2 * i - t - 1] = 0
    F2 = np.zeros_like(F)
    F2[0::2, 0::2] = 1
    F2[1::2, 1::2] = 1
    return F * F2


def _guidedfilter_diagonal(
    I: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray | None = None,
) -> np.ndarray:
    F = _diagonal_window(h, v)

    if N is None:
        N = _imfilter_count(M, F)

    IM = I * M
    pM = p * M
    sum_I, sum_p, sum_Ip, sum_II, sum_pp = _imfilter_many(
        (IM, pM, I * pM, I * IM, p * pM), F, "replicate"
    )
    mean_I = sum_I / N
    mean_p = sum_p / N
    mean_Ip = sum_Ip / N

    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = sum_II / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    dif = (
        sum_II * a * a
        + b * b * N
        + sum_pp
        + 2 * a * b * sum_I
        - 2 * b * sum_p
        - 2 * a * sum_Ip
    )
    dif = np.sqrt(np.maximum(dif / N, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif
    wdif, sum_a, sum_b = _imfilter_many((dif, a * dif, b * dif), F, "replicate")
    mean_a = sum_a / (wdif + 1e-4)
    mean_b = sum_b / (wdif + 1e-4)

    return mean_a * I + mean_b


def _guidedfilter_diagonal_same_guide(
    I: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    M: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    F = _diagonal_window(h, v)
    IM = I * M
    p1M = p1 * M
    p2M = p2 * M
    sum_I, sum_II, sum_p1, sum_Ip1, sum_pp1, sum_p2, sum_Ip2, sum_pp2 = _imfilter_many(
        (IM, I * IM, p1M, I * p1M, p1 * p1M, p2M, I * p2M, p2 * p2M), F, "replicate"
    )
    mean_I = sum_I / N
    mean_II = sum_II / N
    var_I = mean_II - mean_I * mean_I
    denom = var_I + eps

    mean_p1 = sum_p1 / N
    mean_Ip1 = sum_Ip1 / N
    cov_Ip1 = mean_Ip1 - mean_I * mean_p1
    a1 = cov_Ip1 / denom
    b1 = mean_p1 - a1 * mean_I

    mean_p2 = sum_p2 / N
    mean_Ip2 = sum_Ip2 / N
    cov_Ip2 = mean_Ip2 - mean_I * mean_p2
    a2 = cov_Ip2 / denom
    b2 = mean_p2 - a2 * mean_I

    dif1 = sum_II * a1 * a1 + b1 * b1 * N + sum_pp1 + 2 * a1 * b1 * sum_I - 2 * b1 * sum_p1 - 2 * a1 * sum_Ip1
    dif1 = np.sqrt(np.maximum(dif1 / N, 0))
    dif1[dif1 < 1e-3] = 1e-3
    dif1 = 1 / dif1

    dif2 = sum_II * a2 * a2 + b2 * b2 * N + sum_pp2 + 2 * a2 * b2 * sum_I - 2 * b2 * sum_p2 - 2 * a2 * sum_Ip2
    dif2 = np.sqrt(np.maximum(dif2 / N, 0))
    dif2[dif2 < 1e-3] = 1e-3
    dif2 = 1 / dif2

    wdif1, sum_a1, sum_b1, wdif2, sum_a2, sum_b2 = _imfilter_many(
        (dif1, a1 * dif1, b1 * dif1, dif2, a2 * dif2, b2 * dif2), F, "replicate"
    )
    mean_a1 = sum_a1 / (wdif1 + 1e-4)
    mean_b1 = sum_b1 / (wdif1 + 1e-4)
    mean_a2 = sum_a2 / (wdif2 + 1e-4)
    mean_b2 = sum_b2 / (wdif2 + 1e-4)
    out1 = mean_a1 * I + mean_b1
    out2 = mean_a2 * I + mean_b2
    return out1, out2


def _guidedfilter_diagonal_same_guide_from_sums(
    I: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    F: np.ndarray,
    eps: float,
    N: np.ndarray,
    sum_I: np.ndarray,
    sum_II: np.ndarray,
    mean_I: np.ndarray,
    var_I: np.ndarray,
) -> np.ndarray:
    pM = p * M
    sum_p = _imfilter(pM, F, "replicate")
    sum_Ip = _imfilter(I * pM, F, "replicate")
    mean_p = sum_p / N
    mean_Ip = sum_Ip / N
    cov_Ip = mean_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    dif = (
        sum_II * a * a
        + b * b * N
        + _imfilter(p * pM, F, "replicate")
        + 2 * a * b * sum_I
        - 2 * b * sum_p
        - 2 * a * sum_Ip
    )
    dif = np.sqrt(np.maximum(dif / N, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif
    wdif = _imfilter(dif, F, "replicate")
    mean_a = _imfilter(a * dif, F, "replicate") / (wdif + 1e-4)
    mean_b = _imfilter(b * dif, F, "replicate") / (wdif + 1e-4)

    return mean_a * I + mean_b


def _guidedfilter_mlri_diagonal(
    G: np.ndarray,
    R: np.ndarray,
    mask: np.ndarray,
    I: np.ndarray,
    p: np.ndarray,
    M: np.ndarray,
    h: int,
    v: int,
    eps: float,
    N: np.ndarray | None = None,
    N3: np.ndarray | None = None,
) -> np.ndarray:
    F = _diagonal_window(h, v)

    if N is None:
        N = _imfilter_count(M, F)

    IM = I * M
    pM = p * M
    Gm = G * mask
    Rm = R * mask
    sum_Ip, sum_II, sum_G, sum_R, sum_GG, sum_RR, sum_RG = _imfilter_many(
        (I * pM, I * IM, Gm, Rm, G * Gm, R * Rm, R * Gm), F, "replicate"
    )
    mean_Ip = sum_Ip / N
    mean_II = sum_II / N

    a = mean_Ip / (mean_II + eps)
    if N3 is None:
        N3 = _imfilter_count(mask, F)
    mean_G = sum_G / N3
    mean_R = sum_R / N3
    b = mean_R - a * mean_G

    dif = (
        sum_GG * a * a
        + b * b * N3
        + sum_RR
        + 2 * a * b * sum_G
        - 2 * b * sum_R
        - 2 * a * sum_RG
    )
    dif = np.sqrt(np.maximum(dif / N3, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif
    wdif, sum_a, sum_b = _imfilter_many((dif, a * dif, b * dif), F, "replicate")
    mean_a = sum_a / (wdif + 1e-4)
    mean_b = sum_b / (wdif + 1e-4)

    return mean_a * G + mean_b


def _green_interpolation(mosaic: np.ndarray, mask: np.ndarray, pattern: str, eps: float) -> np.ndarray:
    mosaic_r = mosaic[:, :, 0]
    mosaic_g = mosaic[:, :, 1]
    mosaic_b = mosaic[:, :, 2]
    mask_r = mask[:, :, 0]
    mask_g = mask[:, :, 1]
    mask_b = mask[:, :, 2]
    imask_g = ~mask_g
    rawq = mosaic_r + mosaic_g + mosaic_b
    mask_gr, mask_gb = _mask_gr_gb(rawq.shape, pattern)

    Mrh = np.logical_or(mask_r, mask_gr)
    Mbh = np.logical_or(mask_b, mask_gb)
    Mrv = np.logical_or(mask_r, mask_gb)
    Mbv = np.logical_or(mask_b, mask_gr)

    Kh = _ARI_HALF_H
    Kv = _ARI_HALF_V
    rawh = _imfilter(rawq, Kh, "replicate")
    rawv = _imfilter(rawq, Kv, "replicate")

    mosaic_g_gr = mosaic_g * mask_gr
    mosaic_g_gb = mosaic_g * mask_gb
    Guidegrh = mosaic_g_gr + rawh * mask_r
    Guidegbh = mosaic_g_gb + rawh * mask_b
    Guiderh = mosaic_r + rawh * mask_gr
    Guidebh = mosaic_b + rawh * mask_gb
    Guidegrv = mosaic_g_gb + rawv * mask_r
    Guidegbv = mosaic_g_gr + rawv * mask_b
    Guiderv = mosaic_r + rawv * mask_gb
    Guidebv = mosaic_b + rawv * mask_gr

    h = 2
    v = 1
    h2 = 4
    v2 = 0
    itnum = 11

    RI_w2h = np.full(mask_gr.shape, 1e32, dtype=np.float64)
    RI_w2v = np.full(mask_gr.shape, 1e32, dtype=np.float64)
    MLRI_w2h = np.full(mask_gr.shape, 1e32, dtype=np.float64)
    MLRI_w2v = np.full(mask_gr.shape, 1e32, dtype=np.float64)

    RI_Guidegrh = Guidegrh.copy()
    RI_Guidegbh = Guidegbh.copy()
    RI_Guiderh = Guiderh.copy()
    RI_Guidebh = Guidebh.copy()
    RI_Guidegrv = Guidegrv.copy()
    RI_Guidegbv = Guidegbv.copy()
    RI_Guiderv = Guiderv.copy()
    RI_Guidebv = Guidebv.copy()

    MLRI_Guidegrh = Guidegrh.copy()
    MLRI_Guidegbh = Guidegbh.copy()
    MLRI_Guiderh = Guiderh.copy()
    MLRI_Guidebh = Guidebh.copy()
    MLRI_Guidegrv = Guidegrv.copy()
    MLRI_Guidegbv = Guidegbv.copy()
    MLRI_Guiderv = Guiderv.copy()
    MLRI_Guidebv = Guidebv.copy()

    RI_Gh = Guidegrh + Guidegbh
    RI_Gv = Guidegrv + Guidegbv
    MLRI_Gh = Guidegrh + Guidegbh
    MLRI_Gv = Guidegrv + Guidegbv

    Fh_mlri = _ARI_DETAIL_H
    Fv_mlri = _ARI_DETAIL_V
    Kh_residual = _ARI_GREEN_RESIDUAL_H
    Kv_residual = _ARI_GREEN_RESIDUAL_V
    Fh_cri = _ARI_CRI_H
    Fv_cri = _ARI_CRI_V
    Fg = _gaussian_kernel((5, 5), 2)

    for _ in range(itnum):
        RI_tentativeGrh, RI_tentativeRh = _guidedfilter_pair(RI_Guiderh, RI_Guidegrh, Mrh, h, v, eps)
        RI_tentativeGbh, RI_tentativeBh = _guidedfilter_pair(RI_Guidebh, RI_Guidegbh, Mbh, h, v, eps)
        RI_tentativeGrv, RI_tentativeRv = _guidedfilter_pair(RI_Guiderv, RI_Guidegrv, Mrv, v, h, eps)
        RI_tentativeGbv, RI_tentativeBv = _guidedfilter_pair(RI_Guidebv, RI_Guidegbv, Mbv, v, h, eps)

        difR, difGr, difB, difGb = _imfilter_many(
            (MLRI_Guiderh, MLRI_Guidegrh, MLRI_Guidebh, MLRI_Guidegbh), Fh_mlri, "replicate"
        )
        N_Mrh = _bayer_pair_count(rawq.shape, pattern, "r", "gr", h2, v2)
        N_Mbh = _bayer_pair_count(rawq.shape, pattern, "b", "gb", h2, v2)
        MLRI_tentativeRh, MLRI_tentativeGrh = _guidedfilter_mlri_pair(
            MLRI_Guidegrh, MLRI_Guiderh, Mrh, difGr, difR, mask_r, mask_gr, h2, v2, eps, N_Mrh, N_Mrh, N_Mrh
        )
        MLRI_tentativeBh, MLRI_tentativeGbh = _guidedfilter_mlri_pair(
            MLRI_Guidegbh, MLRI_Guidebh, Mbh, difGb, difB, mask_b, mask_gb, h2, v2, eps, N_Mbh, N_Mbh, N_Mbh
        )

        difR, difGr, difB, difGb = _imfilter_many(
            (MLRI_Guiderv, MLRI_Guidegrv, MLRI_Guidebv, MLRI_Guidegbv), Fv_mlri, "replicate"
        )
        N_Mrv = _bayer_pair_count(rawq.shape, pattern, "r", "gb", v2, h2)
        N_Mbv = _bayer_pair_count(rawq.shape, pattern, "b", "gr", v2, h2)
        MLRI_tentativeRv, MLRI_tentativeGrv = _guidedfilter_mlri_pair(
            MLRI_Guidegrv, MLRI_Guiderv, Mrv, difGr, difR, mask_r, mask_gb, v2, h2, eps, N_Mrv, N_Mrv, N_Mrv
        )
        MLRI_tentativeBv, MLRI_tentativeGbv = _guidedfilter_mlri_pair(
            MLRI_Guidegbv, MLRI_Guidebv, Mbv, difGb, difB, mask_b, mask_gr, v2, h2, eps, N_Mbv, N_Mbv, N_Mbv
        )

        RI_residualGrh = (mosaic_g - RI_tentativeGrh) * mask_gr
        RI_residualGbh = (mosaic_g - RI_tentativeGbh) * mask_gb
        RI_residualRh = (mosaic_r - RI_tentativeRh) * mask_r
        RI_residualBh = (mosaic_b - RI_tentativeBh) * mask_b
        RI_residualGrv = (mosaic_g - RI_tentativeGrv) * mask_gb
        RI_residualGbv = (mosaic_g - RI_tentativeGbv) * mask_gr
        RI_residualRv = (mosaic_r - RI_tentativeRv) * mask_r
        RI_residualBv = (mosaic_b - RI_tentativeBv) * mask_b
        MLRI_residualGrh = (mosaic_g - MLRI_tentativeGrh) * mask_gr
        MLRI_residualGbh = (mosaic_g - MLRI_tentativeGbh) * mask_gb
        MLRI_residualRh = (mosaic_r - MLRI_tentativeRh) * mask_r
        MLRI_residualBh = (mosaic_b - MLRI_tentativeBh) * mask_b
        MLRI_residualGrv = (mosaic_g - MLRI_tentativeGrv) * mask_gb
        MLRI_residualGbv = (mosaic_g - MLRI_tentativeGbv) * mask_gr
        MLRI_residualRv = (mosaic_r - MLRI_tentativeRv) * mask_r
        MLRI_residualBv = (mosaic_b - MLRI_tentativeBv) * mask_b

        (
            RI_residualGrh,
            RI_residualGbh,
            RI_residualRh,
            RI_residualBh,
            MLRI_residualGrh,
            MLRI_residualGbh,
            MLRI_residualRh,
            MLRI_residualBh,
        ) = _imfilter_many(
            (
                RI_residualGrh,
                RI_residualGbh,
                RI_residualRh,
                RI_residualBh,
                MLRI_residualGrh,
                MLRI_residualGbh,
                MLRI_residualRh,
                MLRI_residualBh,
            ),
            Kh_residual,
            "replicate",
        )
        (
            RI_residualGrv,
            RI_residualGbv,
            RI_residualRv,
            RI_residualBv,
            MLRI_residualGrv,
            MLRI_residualGbv,
            MLRI_residualRv,
            MLRI_residualBv,
        ) = _imfilter_many(
            (
                RI_residualGrv,
                RI_residualGbv,
                RI_residualRv,
                RI_residualBv,
                MLRI_residualGrv,
                MLRI_residualGbv,
                MLRI_residualRv,
                MLRI_residualBv,
            ),
            Kv_residual,
            "replicate",
        )

        RI_Grh = (RI_tentativeGrh + RI_residualGrh) * mask_r
        RI_Gbh = (RI_tentativeGbh + RI_residualGbh) * mask_b
        RI_Rh = (RI_tentativeRh + RI_residualRh) * mask_gr
        RI_Bh = (RI_tentativeBh + RI_residualBh) * mask_gb
        RI_Grv = (RI_tentativeGrv + RI_residualGrv) * mask_r
        RI_Gbv = (RI_tentativeGbv + RI_residualGbv) * mask_b
        RI_Rv = (RI_tentativeRv + RI_residualRv) * mask_gb
        RI_Bv = (RI_tentativeBv + RI_residualBv) * mask_gr
        MLRI_Grh = (MLRI_tentativeGrh + MLRI_residualGrh) * mask_r
        MLRI_Gbh = (MLRI_tentativeGbh + MLRI_residualGbh) * mask_b
        MLRI_Rh = (MLRI_tentativeRh + MLRI_residualRh) * mask_gr
        MLRI_Bh = (MLRI_tentativeBh + MLRI_residualBh) * mask_gb
        MLRI_Grv = (MLRI_tentativeGrv + MLRI_residualGrv) * mask_r
        MLRI_Gbv = (MLRI_tentativeGbv + MLRI_residualGbv) * mask_b
        MLRI_Rv = (MLRI_tentativeRv + MLRI_residualRv) * mask_gr
        MLRI_Bv = (MLRI_tentativeBv + MLRI_residualBv) * mask_gr

        RI_criGrh = (RI_Guidegrh - RI_tentativeGrh) * Mrh
        RI_criGbh = (RI_Guidegbh - RI_tentativeGbh) * Mbh
        RI_criRh = (RI_Guiderh - RI_tentativeRh) * Mrh
        RI_criBh = (RI_Guidebh - RI_tentativeBh) * Mbh
        RI_criGrv = (RI_Guidegrv - RI_tentativeGrv) * Mrv
        RI_criGbv = (RI_Guidegbv - RI_tentativeGbv) * Mbv
        RI_criRv = (RI_Guiderv - RI_tentativeRv) * Mrv
        RI_criBv = (RI_Guidebv - RI_tentativeBv) * Mbv
        MLRI_criGrh = (MLRI_Guidegrh - MLRI_tentativeGrh) * Mrh
        MLRI_criGbh = (MLRI_Guidegbh - MLRI_tentativeGbh) * Mbh
        MLRI_criRh = (MLRI_Guiderh - MLRI_tentativeRh) * Mrh
        MLRI_criBh = (MLRI_Guidebh - MLRI_tentativeBh) * Mbh
        MLRI_criGrv = (MLRI_Guidegrv - MLRI_tentativeGrv) * Mrv
        MLRI_criGbv = (MLRI_Guidegbv - MLRI_tentativeGbv) * Mbv
        MLRI_criRv = (MLRI_Guiderv - MLRI_tentativeRv) * Mrv
        MLRI_criBv = (MLRI_Guidebv - MLRI_tentativeBv) * Mbv

        (
            RI_difcriGrh,
            RI_difcriGbh,
            RI_difcriRh,
            RI_difcriBh,
            MLRI_difcriGrh,
            MLRI_difcriGbh,
            MLRI_difcriRh,
            MLRI_difcriBh,
        ) = _imfilter_many(
            (RI_criGrh, RI_criGbh, RI_criRh, RI_criBh, MLRI_criGrh, MLRI_criGbh, MLRI_criRh, MLRI_criBh),
            Fh_cri,
            "replicate",
        )
        RI_difcriGrh = np.abs(RI_difcriGrh)
        RI_difcriGbh = np.abs(RI_difcriGbh)
        RI_difcriRh = np.abs(RI_difcriRh)
        RI_difcriBh = np.abs(RI_difcriBh)
        MLRI_difcriGrh = np.abs(MLRI_difcriGrh)
        MLRI_difcriGbh = np.abs(MLRI_difcriGbh)
        MLRI_difcriRh = np.abs(MLRI_difcriRh)
        MLRI_difcriBh = np.abs(MLRI_difcriBh)
        (
            RI_difcriGrv,
            RI_difcriGbv,
            RI_difcriRv,
            RI_difcriBv,
            MLRI_difcriGrv,
            MLRI_difcriGbv,
            MLRI_difcriRv,
            MLRI_difcriBv,
        ) = _imfilter_many(
            (RI_criGrv, RI_criGbv, RI_criRv, RI_criBv, MLRI_criGrv, MLRI_criGbv, MLRI_criRv, MLRI_criBv),
            Fv_cri,
            "replicate",
        )
        RI_difcriGrv = np.abs(RI_difcriGrv)
        RI_difcriGbv = np.abs(RI_difcriGbv)
        RI_difcriRv = np.abs(RI_difcriRv)
        RI_difcriBv = np.abs(RI_difcriBv)
        MLRI_difcriGrv = np.abs(MLRI_difcriGrv)
        MLRI_difcriGbv = np.abs(MLRI_difcriGbv)
        MLRI_difcriRv = np.abs(MLRI_difcriRv)
        MLRI_difcriBv = np.abs(MLRI_difcriBv)

        RI_criGrh = np.abs(RI_criGrh)
        RI_criGbh = np.abs(RI_criGbh)
        RI_criRh = np.abs(RI_criRh)
        RI_criBh = np.abs(RI_criBh)
        RI_criGrv = np.abs(RI_criGrv)
        RI_criGbv = np.abs(RI_criGbv)
        RI_criRv = np.abs(RI_criRv)
        RI_criBv = np.abs(RI_criBv)
        MLRI_criGrh = np.abs(MLRI_criGrh)
        MLRI_criGbh = np.abs(MLRI_criGbh)
        MLRI_criRh = np.abs(MLRI_criRh)
        MLRI_criBh = np.abs(MLRI_criBh)
        MLRI_criGrv = np.abs(MLRI_criGrv)
        MLRI_criGbv = np.abs(MLRI_criGbv)
        MLRI_criRv = np.abs(MLRI_criRv)
        MLRI_criBv = np.abs(MLRI_criBv)

        RI_criGRh = (RI_criGrh + RI_criRh) * Mrh
        RI_criGBh = (RI_criGbh + RI_criBh) * Mbh
        RI_criGRv = (RI_criGrv + RI_criRv) * Mrv
        RI_criGBv = (RI_criGbv + RI_criBv) * Mbv
        MLRI_criGRh = (MLRI_criGrh + MLRI_criRh) * Mrh
        MLRI_criGBh = (MLRI_criGbh + MLRI_criBh) * Mbh
        MLRI_criGRv = (MLRI_criGrv + MLRI_criRv) * Mrv
        MLRI_criGBv = (MLRI_criGbv + MLRI_criBv) * Mbv

        RI_difcriGRh = (RI_difcriGrh + RI_difcriRh) * Mrh
        RI_difcriGBh = (RI_difcriGbh + RI_difcriBh) * Mbh
        RI_difcriGRv = (RI_difcriGrv + RI_difcriRv) * Mrv
        RI_difcriGBv = (RI_difcriGbv + RI_difcriBv) * Mbv
        MLRI_difcriGRh = (MLRI_difcriGrh + MLRI_difcriRh) * Mrh
        MLRI_difcriGBh = (MLRI_difcriGbh + MLRI_difcriBh) * Mbh
        MLRI_difcriGRv = (MLRI_difcriGrv + MLRI_difcriRv) * Mrv
        MLRI_difcriGBv = (MLRI_difcriGbv + MLRI_difcriBv) * Mbv

        RI_crih = RI_criGRh + RI_criGBh
        RI_criv = RI_criGRv + RI_criGBv
        MLRI_crih = MLRI_criGRh + MLRI_criGBh
        MLRI_criv = MLRI_criGRv + MLRI_criGBv
        RI_difcrih = RI_difcriGRh + RI_difcriGBh
        RI_difcriv = RI_difcriGRv + RI_difcriGBv
        MLRI_difcrih = MLRI_difcriGRh + MLRI_difcriGBh
        MLRI_difcriv = MLRI_difcriGRv + MLRI_difcriGBv

        RI_crih, MLRI_crih, RI_difcrih, MLRI_difcrih, RI_criv, MLRI_criv, RI_difcriv, MLRI_difcriv = _imfilter_many(
            (RI_crih, MLRI_crih, RI_difcrih, MLRI_difcrih, RI_criv, MLRI_criv, RI_difcriv, MLRI_difcriv),
            Fg,
            "replicate",
        )

        RI_wh = RI_crih**2 * RI_difcrih
        RI_wv = RI_criv**2 * RI_difcriv
        MLRI_wh = MLRI_crih**2 * MLRI_difcrih
        MLRI_wv = MLRI_criv**2 * MLRI_difcriv

        RI_pih = RI_wh < RI_w2h
        RI_piv = RI_wv < RI_w2v
        MLRI_pih = MLRI_wh < MLRI_w2h
        MLRI_piv = MLRI_wv < MLRI_w2v

        RI_Guidegrh = mosaic_g_gr + RI_Grh
        RI_Guidegbh = mosaic_g_gb + RI_Gbh
        RI_Guidegh = RI_Guidegrh + RI_Guidegbh
        RI_Guiderh = mosaic_r + RI_Rh
        RI_Guidebh = mosaic_b + RI_Bh
        RI_Guidegrv = mosaic_g_gb + RI_Grv
        RI_Guidegbv = mosaic_g_gr + RI_Gbv
        RI_Guidegv = RI_Guidegrv + RI_Guidegbv
        RI_Guiderv = mosaic_r + RI_Rv
        RI_Guidebv = mosaic_b + RI_Bv
        MLRI_Guidegrh = mosaic_g_gr + MLRI_Grh
        MLRI_Guidegbh = mosaic_g_gb + MLRI_Gbh
        MLRI_Guidegh = MLRI_Guidegrh + MLRI_Guidegbh
        MLRI_Guiderh = mosaic_r + MLRI_Rh
        MLRI_Guidebh = mosaic_b + MLRI_Bh
        MLRI_Guidegrv = mosaic_g_gb + MLRI_Grv
        MLRI_Guidegbv = mosaic_g_gr + MLRI_Gbv
        MLRI_Guidegv = MLRI_Guidegrv + MLRI_Guidegbv
        MLRI_Guiderv = mosaic_r + MLRI_Rv
        MLRI_Guidebv = mosaic_b + MLRI_Bv

        RI_Gh[RI_pih] = RI_Guidegh[RI_pih]
        MLRI_Gh[MLRI_pih] = MLRI_Guidegh[MLRI_pih]
        RI_Gv[RI_piv] = RI_Guidegv[RI_piv]
        MLRI_Gv[MLRI_piv] = MLRI_Guidegv[MLRI_piv]

        RI_w2h[RI_pih] = RI_wh[RI_pih]
        RI_w2v[RI_piv] = RI_wv[RI_piv]
        MLRI_w2h[MLRI_pih] = MLRI_wh[MLRI_pih]
        MLRI_w2v[MLRI_piv] = MLRI_wv[MLRI_piv]

        h += 1
        v += 1
        h2 += 1
        v2 += 1

    RI_w2h = 1 / (RI_w2h + 1e-10)
    RI_w2v = 1 / (RI_w2v + 1e-10)
    MLRI_w2h = 1 / (MLRI_w2h + 1e-10)
    MLRI_w2v = 1 / (MLRI_w2v + 1e-10)
    w = RI_w2h + RI_w2v + MLRI_w2h + MLRI_w2v
    green = (RI_w2h * RI_Gh + RI_w2v * RI_Gv + MLRI_w2h * MLRI_Gh + MLRI_w2v * MLRI_Gv) / (w + 1e-32)
    green = green * imask_g + mosaic_g
    return _clip(green, 0, 255)


@lru_cache(maxsize=1)
def _bicubic_residual_kernel() -> np.ndarray:
    s_32 = _s(3 / 2)
    s_12 = _s(1 / 2)
    s_0 = _s(0)
    s_1 = _s(1)
    a = s_32**2
    b = s_32 * s_12
    c = s_12**2
    d = s_0 * s_12
    e = s_0 * s_32
    f = s_1 * s_12
    g = s_1 * s_32
    kernel = np.array(
        [
            [a, g, b, e, b, g, a],
            [g, 0, f, 0, f, 0, g],
            [b, f, c, d, c, f, b],
            [e, 0, d, 1, d, 0, e],
            [b, f, c, d, c, f, b],
            [g, 0, f, 0, f, 0, g],
            [a, g, b, e, b, g, a],
        ],
        dtype=np.float64,
    )
    kernel.setflags(write=False)
    return kernel


def _red_interpolation(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, pattern: str, h: int, v: int, eps: float) -> np.ndarray:
    F = _ARI_LAPLACIAN
    lap_red = _imfilter(mosaic[:, :, 0], F, "replicate")
    lap_green = _imfilter(green * mask[:, :, 0], F, "replicate")

    mask_red = mask[:, :, 0]
    N_red = _bayer_count(green.shape, pattern, "r", h, v)
    tentativeR = _guidedfilter_mlri(green, mosaic[:, :, 0], mask_red, lap_green, lap_red, mask_red, h, v, eps, N=N_red, N3=N_red)
    tentativeR = _clip(tentativeR, 0, 255)
    residualR = mask_red * (mosaic[:, :, 0] - tentativeR)
    residualR = _imfilter(residualR, _bicubic_residual_kernel(), "replicate")
    return _clip(residualR + tentativeR, 0, 255)


def _blue_interpolation(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, pattern: str, h: int, v: int, eps: float) -> np.ndarray:
    F = _ARI_LAPLACIAN
    lap_blue = _imfilter(mosaic[:, :, 2], F, "replicate")
    lap_green = _imfilter(green * mask[:, :, 2], F, "replicate")

    mask_blue = mask[:, :, 2]
    N_blue = _bayer_count(green.shape, pattern, "b", h, v)
    tentativeB = _guidedfilter_mlri(green, mosaic[:, :, 2], mask_blue, lap_green, lap_blue, mask_blue, h, v, eps, N=N_blue, N3=N_blue)
    tentativeB = _clip(tentativeB, 0, 255)
    residualB = mask_blue * (mosaic[:, :, 2] - tentativeB)
    residualB = _imfilter(residualB, _bicubic_residual_kernel(), "replicate")
    return _clip(residualB + tentativeB, 0, 255)


def _inverse_mask(mask: np.ndarray) -> np.ndarray:
    return ~mask


def _red_blue_interpolation_first(
    green: np.ndarray,
    mosaic: np.ndarray,
    mask: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    imask = _inverse_mask(mask)

    f1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64) / 2
    guider1 = mosaic[:, :, 0] + _imfilter(mosaic[:, :, 0], f1, "replicate") * mask[:, :, 2]
    guideg1 = green * imask[:, :, 1]
    guideb1 = mosaic[:, :, 2] + _imfilter(mosaic[:, :, 2], f1, "replicate") * mask[:, :, 0]
    f2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64) / 2
    guider2 = mosaic[:, :, 0] + _imfilter(mosaic[:, :, 0], f2, "replicate") * mask[:, :, 2]
    guideg2 = green * imask[:, :, 1]
    guideb2 = mosaic[:, :, 2] + _imfilter(mosaic[:, :, 2], f2, "replicate") * mask[:, :, 0]

    h, v = 2, 2
    h2, v2 = 2, 0
    itnum = 2

    shape = mask.shape[:2]
    RI_w2R1 = np.full(shape, 1e32, dtype=np.float64)
    RI_w2R2 = np.full(shape, 1e32, dtype=np.float64)
    MLRI_w2R1 = np.full(shape, 1e32, dtype=np.float64)
    MLRI_w2R2 = np.full(shape, 1e32, dtype=np.float64)
    RI_w2B1 = np.full(shape, 1e32, dtype=np.float64)
    RI_w2B2 = np.full(shape, 1e32, dtype=np.float64)
    MLRI_w2B1 = np.full(shape, 1e32, dtype=np.float64)
    MLRI_w2B2 = np.full(shape, 1e32, dtype=np.float64)

    RI_Guideg1 = guideg1.copy()
    RI_Guider1 = guider1.copy()
    RI_Guideb1 = guideb1.copy()
    RI_Guideg2 = guideg2.copy()
    RI_Guider2 = guider2.copy()
    RI_Guideb2 = guideb2.copy()
    MLRI_Guideg1 = guideg1.copy()
    MLRI_Guider1 = guider1.copy()
    MLRI_Guideb1 = guideb1.copy()
    MLRI_Guideg2 = guideg2.copy()
    MLRI_Guider2 = guider2.copy()
    MLRI_Guideb2 = guideb2.copy()

    RI_R1 = guider1.copy()
    RI_R2 = guider2.copy()
    MLRI_R1 = guider1.copy()
    MLRI_R2 = guider2.copy()
    RI_B1 = guideb1.copy()
    RI_B2 = guideb2.copy()
    MLRI_B1 = guideb1.copy()
    MLRI_B2 = guideb2.copy()

    for _ in range(itnum):
        RI_tentativeR1 = _guidedfilter_diagonal(RI_Guideg1, RI_Guider1, imask[:, :, 1], h, v, eps)
        RI_tentativeR2 = _guidedfilter_diagonal(RI_Guideg2, RI_Guider2, imask[:, :, 1], v, h, eps)
        RI_tentativeB1 = _guidedfilter_diagonal(RI_Guideg1, RI_Guideb1, imask[:, :, 1], h, v, eps)
        RI_tentativeB2 = _guidedfilter_diagonal(RI_Guideg2, RI_Guideb2, imask[:, :, 1], v, h, eps)

        f1 = np.array(
            [[-1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, -1]],
            dtype=np.float64,
        )
        difR = _imfilter(MLRI_Guider1, f1, "replicate")
        difG = _imfilter(MLRI_Guideg1, f1, "replicate")
        difB = _imfilter(MLRI_Guideb1, f1, "replicate")
        MLRI_tentativeR1 = _guidedfilter_mlri_diagonal(
            MLRI_Guideg1, MLRI_Guider1, imask[:, :, 1], difG, difR, mask[:, :, 0], h2, v2, eps
        )
        MLRI_tentativeB1 = _guidedfilter_mlri_diagonal(
            MLRI_Guideg1, MLRI_Guideb1, imask[:, :, 1], difG, difB, mask[:, :, 2], h2, v2, eps
        )

        f2 = np.array(
            [[0, 0, 0, 0, -1], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0]],
            dtype=np.float64,
        )
        difR = _imfilter(MLRI_Guider2, f2, "replicate")
        difG = _imfilter(MLRI_Guideg2, f2, "replicate")
        difB = _imfilter(MLRI_Guideb2, f2, "replicate")
        MLRI_tentativeR2 = _guidedfilter_mlri_diagonal(
            MLRI_Guideg2, MLRI_Guider2, imask[:, :, 1], difG, difR, mask[:, :, 0], v2, h2, eps
        )
        MLRI_tentativeB2 = _guidedfilter_mlri_diagonal(
            MLRI_Guideg2, MLRI_Guideb2, imask[:, :, 1], difG, difB, mask[:, :, 2], v2, h2, eps
        )

        RI_residualR1 = (mosaic_r - RI_tentativeR1) * mask_r
        RI_residualB1 = (mosaic_b - RI_tentativeB1) * mask_b
        RI_residualR2 = (mosaic_r - RI_tentativeR2) * mask_r
        RI_residualB2 = (mosaic_b - RI_tentativeB2) * mask_b
        MLRI_residualR1 = (mosaic_r - MLRI_tentativeR1) * mask_r
        MLRI_residualB1 = (mosaic_b - MLRI_tentativeB1) * mask_b
        MLRI_residualR2 = (mosaic_r - MLRI_tentativeR2) * mask_r
        MLRI_residualB2 = (mosaic_b - MLRI_tentativeB2) * mask_b

        k1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64) / 2
        RI_residualR1 = _imfilter(RI_residualR1, k1, "replicate")
        RI_residualB1 = _imfilter(RI_residualB1, k1, "replicate")
        MLRI_residualR1 = _imfilter(MLRI_residualR1, k1, "replicate")
        MLRI_residualB1 = _imfilter(MLRI_residualB1, k1, "replicate")
        k2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64) / 2
        RI_residualR2 = _imfilter(RI_residualR2, k2, "replicate")
        RI_residualB2 = _imfilter(RI_residualB2, k2, "replicate")
        MLRI_residualR2 = _imfilter(MLRI_residualR2, k2, "replicate")
        MLRI_residualB2 = _imfilter(MLRI_residualB2, k2, "replicate")

        RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask_b
        RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask_r
        RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask_b
        RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask_r
        MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask_b
        MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask_r
        MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask_b
        MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask_r

        RI_criR1 = (RI_Guider1 - RI_tentativeR1) * imask_g
        RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * imask_g
        RI_criR2 = (RI_Guider2 - RI_tentativeR2) * imask_g
        RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * imask_g
        MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * imask_g
        MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * imask_g
        MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * imask_g
        MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * imask_g

        f1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.float64)
        RI_difcriR1 = np.abs(_imfilter(RI_criR1, f1, "replicate"))
        RI_difcriB1 = np.abs(_imfilter(RI_criB1, f1, "replicate"))
        MLRI_difcriR1 = np.abs(_imfilter(MLRI_criR1, f1, "replicate"))
        MLRI_difcriB1 = np.abs(_imfilter(MLRI_criB1, f1, "replicate"))
        f2 = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=np.float64)
        RI_difcriR2 = np.abs(_imfilter(RI_criR2, f2, "replicate"))
        RI_difcriB2 = np.abs(_imfilter(RI_criB2, f2, "replicate"))
        MLRI_difcriR2 = np.abs(_imfilter(MLRI_criR2, f2, "replicate"))
        MLRI_difcriB2 = np.abs(_imfilter(MLRI_criB2, f2, "replicate"))

        RI_criR1 = np.abs(RI_criR1)
        RI_criB1 = np.abs(RI_criB1)
        RI_criR2 = np.abs(RI_criR2)
        RI_criB2 = np.abs(RI_criB2)
        MLRI_criR1 = np.abs(MLRI_criR1)
        MLRI_criB1 = np.abs(MLRI_criB1)
        MLRI_criR2 = np.abs(MLRI_criR2)
        MLRI_criB2 = np.abs(MLRI_criB2)

        RI_criR1 = RI_criR1 + RI_criB1
        RI_criB1 = RI_criB1 + RI_criR1
        RI_criR2 = RI_criR2 + RI_criB2
        RI_criB2 = RI_criB2 + RI_criR2
        MLRI_criR1 = MLRI_criR1 + MLRI_criB1
        MLRI_criB1 = MLRI_criB1 + MLRI_criR1
        MLRI_criR2 = MLRI_criR2 + MLRI_criB2
        MLRI_criB2 = MLRI_criB2 + MLRI_criR2

        RI_difcriR1 = RI_difcriR1 + RI_difcriB1
        RI_difcriB1 = RI_difcriB1 + RI_difcriR1
        RI_difcriR2 = RI_difcriR2 + RI_difcriB2
        RI_difcriB2 = RI_difcriB2 + RI_difcriR2
        MLRI_difcriR1 = MLRI_difcriR1 + MLRI_difcriB1
        MLRI_difcriB1 = MLRI_difcriB1 + MLRI_difcriR1
        MLRI_difcriR2 = MLRI_difcriR2 + MLRI_difcriB2
        MLRI_difcriB2 = MLRI_difcriB2 + MLRI_difcriR2

        fg = _gaussian_kernel((5, 5), 2)
        m1 = _imfilter(imask[:, :, 1], fg, "replicate")
        RI_criR1 = _imfilter(RI_criR1, fg, "replicate") / m1 * imask[:, :, 1]
        MLRI_criR1 = _imfilter(MLRI_criR1, fg, "replicate") / m1 * imask[:, :, 1]
        RI_criB1 = _imfilter(RI_criB1, fg, "replicate") / m1 * imask[:, :, 1]
        MLRI_criB1 = _imfilter(MLRI_criB1, fg, "replicate") / m1 * imask[:, :, 1]
        RI_difcriR1 = _imfilter(RI_difcriR1, fg, "replicate") / m1 * imask[:, :, 1]
        MLRI_difcriR1 = _imfilter(MLRI_difcriR1, fg, "replicate") / m1 * imask[:, :, 1]
        RI_difcriB1 = _imfilter(RI_difcriB1, fg, "replicate") / m1 * imask[:, :, 1]
        MLRI_difcriB1 = _imfilter(MLRI_difcriB1, fg, "replicate") / m1 * imask[:, :, 1]

        m2 = _imfilter(imask[:, :, 1], fg, "replicate")
        RI_criR2 = _imfilter(RI_criR2, fg, "replicate") / m2 * imask[:, :, 1]
        MLRI_criR2 = _imfilter(MLRI_criR2, fg, "replicate") / m2 * imask[:, :, 1]
        RI_criB2 = _imfilter(RI_criB2, fg, "replicate") / m2 * imask[:, :, 1]
        MLRI_criB2 = _imfilter(MLRI_criB2, fg, "replicate") / m2 * imask[:, :, 1]
        RI_difcriR2 = _imfilter(RI_difcriR2, fg, "replicate") / m2 * imask[:, :, 1]
        MLRI_difcriR2 = _imfilter(MLRI_difcriR2, fg, "replicate") / m2 * imask[:, :, 1]
        RI_difcriB2 = _imfilter(RI_difcriB2, fg, "replicate") / m2 * imask[:, :, 1]
        MLRI_difcriB2 = _imfilter(MLRI_difcriB2, fg, "replicate") / m2 * imask[:, :, 1]

        RI_wR1 = RI_criR1**2 * RI_difcriR1
        RI_wR2 = RI_criR2**2 * RI_difcriR2
        MLRI_wR1 = MLRI_criR1**2 * MLRI_difcriR1
        MLRI_wR2 = MLRI_criR2**2 * MLRI_difcriR2
        RI_wB1 = RI_criB1**2 * RI_difcriB1
        RI_wB2 = RI_criB2**2 * RI_difcriB2
        MLRI_wB1 = MLRI_criB1**2 * MLRI_difcriB1
        MLRI_wB2 = MLRI_criB2**2 * MLRI_difcriB2

        RI_piR1 = RI_wR1 < RI_w2R1
        RI_piR2 = RI_wR2 < RI_w2R2
        MLRI_piR1 = MLRI_wR1 < MLRI_w2R1
        MLRI_piR2 = MLRI_wR2 < MLRI_w2R2
        RI_piB1 = RI_wB1 < RI_w2B1
        RI_piB2 = RI_wB2 < RI_w2B2
        MLRI_piB1 = MLRI_wB1 < MLRI_w2B1
        MLRI_piB2 = MLRI_wB2 < MLRI_w2B2

        RI_Guider1 = mosaic[:, :, 0] + RI_R1
        RI_Guideb1 = mosaic[:, :, 2] + RI_B1
        RI_Guider2 = mosaic[:, :, 0] + RI_R2
        RI_Guideb2 = mosaic[:, :, 2] + RI_B2
        MLRI_Guider1 = mosaic[:, :, 0] + MLRI_R1
        MLRI_Guideb1 = mosaic[:, :, 2] + MLRI_B1
        MLRI_Guider2 = mosaic[:, :, 0] + MLRI_R2
        MLRI_Guideb2 = mosaic[:, :, 2] + MLRI_B2

        RI_R1[RI_piR1] = RI_Guider1[RI_piR1]
        MLRI_R1[MLRI_piR1] = MLRI_Guider1[MLRI_piR1]
        RI_R2[RI_piR2] = RI_Guider2[RI_piR2]
        MLRI_R2[MLRI_piR2] = MLRI_Guider2[MLRI_piR2]
        RI_B1[RI_piB1] = RI_Guideb1[RI_piB1]
        MLRI_B1[MLRI_piB1] = MLRI_Guideb1[MLRI_piB1]
        RI_B2[RI_piB2] = RI_Guideb2[RI_piB2]
        MLRI_B2[MLRI_piB2] = MLRI_Guideb2[MLRI_piB2]

        RI_w2R1[RI_piR1] = RI_wR1[RI_piR1]
        RI_w2R2[RI_piR2] = RI_wR2[RI_piR2]
        RI_w2B1[RI_piB1] = RI_wB1[RI_piB1]
        RI_w2B2[RI_piB2] = RI_wB2[RI_piB2]
        MLRI_w2R1[MLRI_piR1] = MLRI_wR1[MLRI_piR1]
        MLRI_w2R2[MLRI_piR2] = MLRI_wR2[MLRI_piR2]
        MLRI_w2B1[MLRI_piB1] = MLRI_wB1[MLRI_piB1]
        MLRI_w2B2[MLRI_piB2] = MLRI_wB2[MLRI_piB2]

        h += 1
        v += 1
        h2 += 1
        v2 += 1

    RI_w2R1 = 1 / (RI_w2R1 + 1e-10)
    RI_w2R2 = 1 / (RI_w2R2 + 1e-10)
    MLRI_w2R1 = 1 / (MLRI_w2R1 + 1e-10)
    MLRI_w2R2 = 1 / (MLRI_w2R2 + 1e-10)
    RI_w2B1 = 1 / (RI_w2B1 + 1e-10)
    RI_w2B2 = 1 / (RI_w2B2 + 1e-10)
    MLRI_w2B1 = 1 / (MLRI_w2B1 + 1e-10)
    MLRI_w2B2 = 1 / (MLRI_w2B2 + 1e-10)

    wR = RI_w2R1 + RI_w2R2 + MLRI_w2R1 + MLRI_w2R2
    wB = RI_w2B1 + RI_w2B2 + MLRI_w2B1 + MLRI_w2B2
    red = (RI_w2R1 * RI_R1 + RI_w2R2 * RI_R2 + MLRI_w2R1 * MLRI_R1 + MLRI_w2R2 * MLRI_R2) / (wR + 1e-32)
    blue = (RI_w2B1 * RI_B1 + RI_w2B2 * RI_B2 + MLRI_w2B1 * MLRI_B1 + MLRI_w2B2 * MLRI_B2) / (wB + 1e-32)

    red = red * mask[:, :, 2] + mosaic[:, :, 0]
    blue = blue * mask[:, :, 0] + mosaic[:, :, 2]
    return _clip(red, 0, 255), _clip(blue, 0, 255)


def _red_blue_interpolation_second(
    green: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    mask: np.ndarray,
    pattern: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    imask = _inverse_mask(mask)

    f1 = np.array([[0.5, 0, 0.5]], dtype=np.float64)
    guider1 = red + _imfilter(red, f1, "replicate") * mask[:, :, 1]
    guideg1 = green
    guideb1 = blue + _imfilter(blue, f1, "replicate") * mask[:, :, 1]
    f2 = f1.T
    guider2 = red + _imfilter(red, f2, "replicate") * mask[:, :, 1]
    guideg2 = green
    guideb2 = blue + _imfilter(blue, f2, "replicate") * mask[:, :, 1]

    h, v = 2, 2
    h2, v2 = 2, 0
    itnum = 2
    shape = mask[:, :, 0].shape
    RI_w2R1 = np.ones(shape) * 1e32
    RI_w2R2 = np.ones(shape) * 1e32
    MLRI_w2R1 = np.ones(shape) * 1e32
    MLRI_w2R2 = np.ones(shape) * 1e32
    RI_w2B1 = np.ones(shape) * 1e32
    RI_w2B2 = np.ones(shape) * 1e32
    MLRI_w2B1 = np.ones(shape) * 1e32
    MLRI_w2B2 = np.ones(shape) * 1e32

    RI_Guideg1 = guideg1.copy()
    RI_Guider1 = guider1.copy()
    RI_Guideb1 = guideb1.copy()
    RI_Guideg2 = guideg2.copy()
    RI_Guider2 = guider2.copy()
    RI_Guideb2 = guideb2.copy()
    MLRI_Guideg1 = guideg1.copy()
    MLRI_Guider1 = guider1.copy()
    MLRI_Guideb1 = guideb1.copy()
    MLRI_Guideg2 = guideg2.copy()
    MLRI_Guider2 = guider2.copy()
    MLRI_Guideb2 = guideb2.copy()

    RI_R1 = guider1.copy()
    RI_R2 = guider2.copy()
    MLRI_R1 = guider1.copy()
    MLRI_R2 = guider2.copy()
    RI_B1 = guideb1.copy()
    RI_B2 = guideb2.copy()
    MLRI_B1 = guideb1.copy()
    MLRI_B2 = guideb2.copy()

    for _ in range(itnum):
        m = np.ones(shape, dtype=np.float64)
        RI_tentativeR1 = _guidedfilter(RI_Guideg1, RI_Guider1, m, h, v, eps)
        RI_tentativeB1 = _guidedfilter(RI_Guideg1, RI_Guideb1, m, h, v, eps)
        RI_tentativeR2 = _guidedfilter(RI_Guideg2, RI_Guider2, m, v, h, eps)
        RI_tentativeB2 = _guidedfilter(RI_Guideg2, RI_Guideb2, m, v, h, eps)

        f1 = np.array([[-1, 0, 2, 0, -1]], dtype=np.float64)
        difR = _imfilter(MLRI_Guider1, f1, "replicate")
        difG = _imfilter(MLRI_Guideg1, f1, "replicate")
        difB = _imfilter(MLRI_Guideb1, f1, "replicate")
        MLRI_tentativeR1 = _guidedfilter_mlri(MLRI_Guideg1, MLRI_Guider1, m, difG, difR, imask[:, :, 1], h2, v2, eps)
        MLRI_tentativeB1 = _guidedfilter_mlri(MLRI_Guideg1, MLRI_Guideb1, m, difG, difB, imask[:, :, 1], h2, v2, eps)

        f2 = f1.T
        difR = _imfilter(MLRI_Guider2, f2, "replicate")
        difG = _imfilter(MLRI_Guideg2, f2, "replicate")
        difB = _imfilter(MLRI_Guideb2, f2, "replicate")
        MLRI_tentativeR2 = _guidedfilter_mlri(MLRI_Guideg2, MLRI_Guider2, m, difG, difR, imask[:, :, 1], v2, h2, eps)
        MLRI_tentativeB2 = _guidedfilter_mlri(MLRI_Guideg2, MLRI_Guideb2, m, difG, difB, imask[:, :, 1], v2, h2, eps)

        RI_residualR1 = (red - RI_tentativeR1) * imask[:, :, 1]
        RI_residualB1 = (blue - RI_tentativeB1) * imask[:, :, 1]
        RI_residualR2 = (red - RI_tentativeR2) * imask[:, :, 1]
        RI_residualB2 = (blue - RI_tentativeB2) * imask[:, :, 1]
        MLRI_residualR1 = (red - MLRI_tentativeR1) * imask[:, :, 1]
        MLRI_residualB1 = (blue - MLRI_tentativeB1) * imask[:, :, 1]
        MLRI_residualR2 = (red - MLRI_tentativeR2) * imask[:, :, 1]
        MLRI_residualB2 = (blue - MLRI_tentativeB2) * imask[:, :, 1]

        k1 = np.array([[0.5, 0, 0.5]], dtype=np.float64)
        RI_residualR1 = _imfilter(RI_residualR1, k1, "replicate")
        RI_residualB1 = _imfilter(RI_residualB1, k1, "replicate")
        MLRI_residualR1 = _imfilter(MLRI_residualR1, k1, "replicate")
        MLRI_residualB1 = _imfilter(MLRI_residualB1, k1, "replicate")
        k2 = k1.T
        RI_residualR2 = _imfilter(RI_residualR2, k2, "replicate")
        RI_residualB2 = _imfilter(RI_residualB2, k2, "replicate")
        MLRI_residualR2 = _imfilter(MLRI_residualR2, k2, "replicate")
        MLRI_residualB2 = _imfilter(MLRI_residualB2, k2, "replicate")

        RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask[:, :, 1]
        RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask[:, :, 1]
        RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask[:, :, 1]
        RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask[:, :, 1]
        MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask[:, :, 1]
        MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask[:, :, 1]
        MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask[:, :, 1]
        MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask[:, :, 1]

        RI_criR1 = (RI_Guider1 - RI_tentativeR1) * m
        RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * m
        RI_criR2 = (RI_Guider2 - RI_tentativeR2) * m
        RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * m
        MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * m
        MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * m
        MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * m
        MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * m

        f1 = np.array([[-1, 0, 1]], dtype=np.float64)
        RI_difcriR1 = np.abs(_imfilter(RI_criR1, f1, "replicate"))
        RI_difcriB1 = np.abs(_imfilter(RI_criB1, f1, "replicate"))
        MLRI_difcriR1 = np.abs(_imfilter(MLRI_criR1, f1, "replicate"))
        MLRI_difcriB1 = np.abs(_imfilter(MLRI_criB1, f1, "replicate"))
        f2 = f1.T
        RI_difcriR2 = np.abs(_imfilter(RI_criR2, f2, "replicate"))
        RI_difcriB2 = np.abs(_imfilter(RI_criB2, f2, "replicate"))
        MLRI_difcriR2 = np.abs(_imfilter(MLRI_criR2, f2, "replicate"))
        MLRI_difcriB2 = np.abs(_imfilter(MLRI_criB2, f2, "replicate"))

        RI_criR1 = np.abs(RI_criR1)
        RI_criB1 = np.abs(RI_criB1)
        RI_criR2 = np.abs(RI_criR2)
        RI_criB2 = np.abs(RI_criB2)
        MLRI_criR1 = np.abs(MLRI_criR1)
        MLRI_criB1 = np.abs(MLRI_criB1)
        MLRI_criR2 = np.abs(MLRI_criR2)
        MLRI_criB2 = np.abs(MLRI_criB2)

        RI_criR1 = RI_criR1 + RI_criB1
        RI_criB1 = RI_criB1 + RI_criR1
        RI_criR2 = RI_criR2 + RI_criB2
        RI_criB2 = RI_criB2 + RI_criR2
        MLRI_criR1 = MLRI_criR1 + MLRI_criB1
        MLRI_criB1 = MLRI_criB1 + MLRI_criR1
        MLRI_criR2 = MLRI_criR2 + MLRI_criB2
        MLRI_criB2 = MLRI_criB2 + MLRI_criR2

        RI_difcriR1 = RI_difcriR1 + RI_difcriB1
        RI_difcriB1 = RI_difcriB1 + RI_difcriR1
        RI_difcriR2 = RI_difcriR2 + RI_difcriB2
        RI_difcriB2 = RI_difcriB2 + RI_difcriR2
        MLRI_difcriR1 = MLRI_difcriR1 + MLRI_difcriB1
        MLRI_difcriB1 = MLRI_difcriB1 + MLRI_difcriR1
        MLRI_difcriR2 = MLRI_difcriR2 + MLRI_difcriB2
        MLRI_difcriB2 = MLRI_difcriB2 + MLRI_difcriR2

        fg = _gaussian_kernel((5, 5), 2)
        RI_criR1 = _imfilter(RI_criR1, fg, "replicate")
        MLRI_criR1 = _imfilter(MLRI_criR1, fg, "replicate")
        RI_criB1 = _imfilter(RI_criB1, fg, "replicate")
        MLRI_criB1 = _imfilter(MLRI_criB1, fg, "replicate")
        RI_difcriR1 = _imfilter(RI_difcriR1, fg, "replicate")
        MLRI_difcriR1 = _imfilter(MLRI_difcriR1, fg, "replicate")
        RI_difcriB1 = _imfilter(RI_difcriB1, fg, "replicate")
        MLRI_difcriB1 = _imfilter(MLRI_difcriB1, fg, "replicate")
        RI_criR2 = _imfilter(RI_criR2, fg, "replicate")
        MLRI_criR2 = _imfilter(MLRI_criR2, fg, "replicate")
        RI_criB2 = _imfilter(RI_criB2, fg, "replicate")
        MLRI_criB2 = _imfilter(MLRI_criB2, fg, "replicate")
        RI_difcriR2 = _imfilter(RI_difcriR2, fg, "replicate")
        MLRI_difcriR2 = _imfilter(MLRI_difcriR2, fg, "replicate")
        RI_difcriB2 = _imfilter(RI_difcriB2, fg, "replicate")
        MLRI_difcriB2 = _imfilter(MLRI_difcriB2, fg, "replicate")

        RI_wR1 = RI_criR1**2 * RI_difcriR1
        RI_wR2 = RI_criR2**2 * RI_difcriR2
        MLRI_wR1 = MLRI_criR1**2 * MLRI_difcriR1
        MLRI_wR2 = MLRI_criR2**2 * MLRI_difcriR2
        RI_wB1 = RI_criB1**2 * RI_difcriB1
        RI_wB2 = RI_criB2**2 * RI_difcriB2
        MLRI_wB1 = MLRI_criB1**2 * MLRI_difcriB1
        MLRI_wB2 = MLRI_criB2**2 * MLRI_difcriB2

        RI_piR1 = RI_wR1 < RI_w2R1
        RI_piR2 = RI_wR2 < RI_w2R2
        MLRI_piR1 = MLRI_wR1 < MLRI_w2R1
        MLRI_piR2 = MLRI_wR2 < MLRI_w2R2
        RI_piB1 = RI_wB1 < RI_w2B1
        RI_piB2 = RI_wB2 < RI_w2B2
        MLRI_piB1 = MLRI_wB1 < MLRI_w2B1
        MLRI_piB2 = MLRI_wB2 < MLRI_w2B2

        RI_Guider1 = red + RI_R1
        RI_Guideb1 = blue + RI_B1
        RI_Guider2 = red + RI_R2
        RI_Guideb2 = blue + RI_B2
        MLRI_Guider1 = red + MLRI_R1
        MLRI_Guideb1 = blue + MLRI_B1
        MLRI_Guider2 = red + MLRI_R2
        MLRI_Guideb2 = blue + MLRI_B2

        RI_R1[RI_piR1] = RI_Guider1[RI_piR1]
        MLRI_R1[MLRI_piR1] = MLRI_Guider1[MLRI_piR1]
        RI_R2[RI_piR2] = RI_Guider2[RI_piR2]
        MLRI_R2[MLRI_piR2] = MLRI_Guider2[MLRI_piR2]
        RI_B1[RI_piB1] = RI_Guideb1[RI_piB1]
        MLRI_B1[MLRI_piB1] = MLRI_Guideb1[MLRI_piB1]
        RI_B2[RI_piB2] = RI_Guideb2[RI_piB2]
        MLRI_B2[MLRI_piB2] = MLRI_Guideb2[MLRI_piB2]

        RI_w2R1[RI_piR1] = RI_wR1[RI_piR1]
        RI_w2R2[RI_piR2] = RI_wR2[RI_piR2]
        RI_w2B1[RI_piB1] = RI_wB1[RI_piB1]
        RI_w2B2[RI_piB2] = RI_wB2[RI_piB2]
        MLRI_w2R1[MLRI_piR1] = MLRI_wR1[MLRI_piR1]
        MLRI_w2R2[MLRI_piR2] = MLRI_wR2[MLRI_piR2]
        MLRI_w2B1[MLRI_piB1] = MLRI_wB1[MLRI_piB1]
        MLRI_w2B2[MLRI_piB2] = MLRI_wB2[MLRI_piB2]

        h += 1
        v += 1
        h2 += 1
        v2 += 1

    RI_w2R1 = 1 / (RI_w2R1 + 1e-10)
    RI_w2R2 = 1 / (RI_w2R2 + 1e-10)
    MLRI_w2R1 = 1 / (MLRI_w2R1 + 1e-10)
    MLRI_w2R2 = 1 / (MLRI_w2R2 + 1e-10)
    RI_w2B1 = 1 / (RI_w2B1 + 1e-10)
    RI_w2B2 = 1 / (RI_w2B2 + 1e-10)
    MLRI_w2B1 = 1 / (MLRI_w2B1 + 1e-10)
    MLRI_w2B2 = 1 / (MLRI_w2B2 + 1e-10)

    wR = RI_w2R1 + RI_w2R2 + MLRI_w2R1 + MLRI_w2R2
    wB = RI_w2B1 + RI_w2B2 + MLRI_w2B1 + MLRI_w2B2
    red2 = (RI_w2R1 * RI_R1 + RI_w2R2 * RI_R2 + MLRI_w2R1 * MLRI_R1 + MLRI_w2R2 * MLRI_R2) / (wR + 1e-32)
    blue2 = (RI_w2B1 * RI_B1 + RI_w2B2 * RI_B2 + MLRI_w2B1 * MLRI_B1 + MLRI_w2B2 * MLRI_B2) / (wB + 1e-32)

    red = red + red2 * mask[:, :, 1]
    blue = blue + blue2 * mask[:, :, 1]
    return _clip(red, 0, 255), _clip(blue, 0, 255)


def _inverse_mask(mask: np.ndarray) -> np.ndarray:
    return ~mask


def _red_blue_interpolation_first(
    green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, pattern: str, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    mosaic_r = mosaic[:, :, 0]
    mosaic_b = mosaic[:, :, 2]
    mask_r = mask[:, :, 0]
    mask_g = mask[:, :, 1]
    mask_b = mask[:, :, 2]
    imask_g = ~mask_g

    F1 = _ARI_DIAG_HALF_1
    F1_red, F1_blue = _imfilter_many((mosaic_r, mosaic_b), F1, "replicate")
    Guider1 = mosaic_r + F1_red * mask_b
    Guideg1 = green * imask_g
    Guideb1 = mosaic_b + F1_blue * mask_r
    F2 = _ARI_DIAG_HALF_2
    F2_red, F2_blue = _imfilter_many((mosaic_r, mosaic_b), F2, "replicate")
    Guider2 = mosaic_r + F2_red * mask_b
    Guideg2 = green * imask_g
    Guideb2 = mosaic_b + F2_blue * mask_r

    h = 2
    v = 2
    h2 = 2
    v2 = 0
    itnum = 2

    RI_w2R1 = np.full(mask_r.shape, 1e32, dtype=np.float64)
    RI_w2R2 = np.full(mask_r.shape, 1e32, dtype=np.float64)
    MLRI_w2R1 = np.full(mask_r.shape, 1e32, dtype=np.float64)
    MLRI_w2R2 = np.full(mask_r.shape, 1e32, dtype=np.float64)
    RI_w2B1 = np.full(mask_r.shape, 1e32, dtype=np.float64)
    RI_w2B2 = np.full(mask_r.shape, 1e32, dtype=np.float64)
    MLRI_w2B1 = np.full(mask_r.shape, 1e32, dtype=np.float64)
    MLRI_w2B2 = np.full(mask_r.shape, 1e32, dtype=np.float64)

    RI_Guideg1 = Guideg1.copy()
    RI_Guider1 = Guider1.copy()
    RI_Guideb1 = Guideb1.copy()
    RI_Guideg2 = Guideg2.copy()
    RI_Guider2 = Guider2.copy()
    RI_Guideb2 = Guideb2.copy()
    MLRI_Guideg1 = Guideg1.copy()
    MLRI_Guider1 = Guider1.copy()
    MLRI_Guideb1 = Guideb1.copy()
    MLRI_Guideg2 = Guideg2.copy()
    MLRI_Guider2 = Guider2.copy()
    MLRI_Guideb2 = Guideb2.copy()

    RI_R1 = Guider1.copy()
    RI_R2 = Guider2.copy()
    MLRI_R1 = Guider1.copy()
    MLRI_R2 = Guider2.copy()
    RI_B1 = Guideb1.copy()
    RI_B2 = Guideb2.copy()
    MLRI_B1 = Guideb1.copy()
    MLRI_B2 = Guideb2.copy()

    F1_mlri = _ARI_DIAG_DETAIL_1
    F2_mlri = _ARI_DIAG_DETAIL_2
    K1_residual = _ARI_DIAG_HALF_1
    K2_residual = _ARI_DIAG_HALF_2
    F1_cri = _ARI_DIAG_CRI_1
    F2_cri = _ARI_DIAG_CRI_2
    Fg = _gaussian_kernel((5, 5), 2)
    M_smooth = _imfilter(imask_g, Fg, "replicate")

    for _ in range(itnum):
        F_diag1 = _diagonal_window(h, v)
        N_imask_diag1 = _diagonal_bayer_count(imask_g.shape, pattern, ("r", "b"), h, v)
        if h == v:
            F_diag2 = F_diag1
            N_imask_diag2 = N_imask_diag1
        else:
            F_diag2 = _diagonal_window(v, h)
            N_imask_diag2 = _diagonal_bayer_count(imask_g.shape, pattern, ("r", "b"), v, h)
        RI_tentativeR1, RI_tentativeB1 = _guidedfilter_diagonal_same_guide(
            RI_Guideg1, RI_Guider1, RI_Guideb1, imask_g, h, v, eps, N=N_imask_diag1
        )
        RI_tentativeR2, RI_tentativeB2 = _guidedfilter_diagonal_same_guide(
            RI_Guideg2, RI_Guider2, RI_Guideb2, imask_g, v, h, eps, N=N_imask_diag2
        )

        difR, difG, difB = _imfilter_many((MLRI_Guider1, MLRI_Guideg1, MLRI_Guideb1), F1_mlri, "replicate")
        F_mlri_diag1 = _diagonal_window(h2, v2)
        N_mask_r_diag1 = _diagonal_bayer_count(mask_r.shape, pattern, ("r",), h2, v2)
        N_mask_b_diag1 = _diagonal_bayer_count(mask_b.shape, pattern, ("b",), h2, v2)
        N3_imask_diag1 = _diagonal_bayer_count(imask_g.shape, pattern, ("r", "b"), h2, v2)
        MLRI_tentativeR1 = _guidedfilter_mlri_diagonal(
            MLRI_Guideg1, MLRI_Guider1, imask_g, difG, difR, mask_r, h2, v2, eps, N=N_mask_r_diag1, N3=N3_imask_diag1
        )
        MLRI_tentativeB1 = _guidedfilter_mlri_diagonal(
            MLRI_Guideg1, MLRI_Guideb1, imask_g, difG, difB, mask_b, h2, v2, eps, N=N_mask_b_diag1, N3=N3_imask_diag1
        )

        difR, difG, difB = _imfilter_many((MLRI_Guider2, MLRI_Guideg2, MLRI_Guideb2), F2_mlri, "replicate")
        F_mlri_diag2 = _diagonal_window(v2, h2)
        N_mask_r_diag2 = _diagonal_bayer_count(mask_r.shape, pattern, ("r",), v2, h2)
        N_mask_b_diag2 = _diagonal_bayer_count(mask_b.shape, pattern, ("b",), v2, h2)
        N3_imask_diag2 = _diagonal_bayer_count(imask_g.shape, pattern, ("r", "b"), v2, h2)
        MLRI_tentativeR2 = _guidedfilter_mlri_diagonal(
            MLRI_Guideg2, MLRI_Guider2, imask_g, difG, difR, mask_r, v2, h2, eps, N=N_mask_r_diag2, N3=N3_imask_diag2
        )
        MLRI_tentativeB2 = _guidedfilter_mlri_diagonal(
            MLRI_Guideg2, MLRI_Guideb2, imask_g, difG, difB, mask_b, v2, h2, eps, N=N_mask_b_diag2, N3=N3_imask_diag2
        )

        RI_residualR1 = (mosaic_r - RI_tentativeR1) * mask_r
        RI_residualB1 = (mosaic_b - RI_tentativeB1) * mask_b
        RI_residualR2 = (mosaic_r - RI_tentativeR2) * mask_r
        RI_residualB2 = (mosaic_b - RI_tentativeB2) * mask_b
        MLRI_residualR1 = (mosaic_r - MLRI_tentativeR1) * mask_r
        MLRI_residualB1 = (mosaic_b - MLRI_tentativeB1) * mask_b
        MLRI_residualR2 = (mosaic_r - MLRI_tentativeR2) * mask_r
        MLRI_residualB2 = (mosaic_b - MLRI_tentativeB2) * mask_b

        RI_residualR1, RI_residualB1, MLRI_residualR1, MLRI_residualB1 = _imfilter_many(
            (RI_residualR1, RI_residualB1, MLRI_residualR1, MLRI_residualB1), K1_residual, "replicate"
        )
        RI_residualR2, RI_residualB2, MLRI_residualR2, MLRI_residualB2 = _imfilter_many(
            (RI_residualR2, RI_residualB2, MLRI_residualR2, MLRI_residualB2), K2_residual, "replicate"
        )

        RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask_b
        RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask_r
        RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask_b
        RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask_r
        MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask_b
        MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask_r
        MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask_b
        MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask_r

        RI_criR1 = (RI_Guider1 - RI_tentativeR1) * imask_g
        RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * imask_g
        RI_criR2 = (RI_Guider2 - RI_tentativeR2) * imask_g
        RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * imask_g
        MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * imask_g
        MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * imask_g
        MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * imask_g
        MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * imask_g

        RI_difcriR1, RI_difcriB1, MLRI_difcriR1, MLRI_difcriB1 = _imfilter_many(
            (RI_criR1, RI_criB1, MLRI_criR1, MLRI_criB1), F1_cri, "replicate"
        )
        RI_difcriR1 = np.abs(RI_difcriR1)
        RI_difcriB1 = np.abs(RI_difcriB1)
        MLRI_difcriR1 = np.abs(MLRI_difcriR1)
        MLRI_difcriB1 = np.abs(MLRI_difcriB1)
        RI_difcriR2, RI_difcriB2, MLRI_difcriR2, MLRI_difcriB2 = _imfilter_many(
            (RI_criR2, RI_criB2, MLRI_criR2, MLRI_criB2), F2_cri, "replicate"
        )
        RI_difcriR2 = np.abs(RI_difcriR2)
        RI_difcriB2 = np.abs(RI_difcriB2)
        MLRI_difcriR2 = np.abs(MLRI_difcriR2)
        MLRI_difcriB2 = np.abs(MLRI_difcriB2)

        RI_criR1 = np.abs(RI_criR1)
        RI_criB1 = np.abs(RI_criB1)
        RI_criR2 = np.abs(RI_criR2)
        RI_criB2 = np.abs(RI_criB2)
        MLRI_criR1 = np.abs(MLRI_criR1)
        MLRI_criB1 = np.abs(MLRI_criB1)
        MLRI_criR2 = np.abs(MLRI_criR2)
        MLRI_criB2 = np.abs(MLRI_criB2)

        RI_criR1 = RI_criR1 + RI_criB1
        RI_criB1 = RI_criB1 + RI_criR1
        RI_criR2 = RI_criR2 + RI_criB2
        RI_criB2 = RI_criB2 + RI_criR2
        MLRI_criR1 = MLRI_criR1 + MLRI_criB1
        MLRI_criB1 = MLRI_criB1 + MLRI_criR1
        MLRI_criR2 = MLRI_criR2 + MLRI_criB2
        MLRI_criB2 = MLRI_criB2 + MLRI_criR2

        RI_difcriR1 = RI_difcriR1 + RI_difcriB1
        RI_difcriB1 = RI_difcriB1 + RI_difcriR1
        RI_difcriR2 = RI_difcriR2 + RI_difcriB2
        RI_difcriB2 = RI_difcriB2 + RI_difcriR2
        MLRI_difcriR1 = MLRI_difcriR1 + MLRI_difcriB1
        MLRI_difcriB1 = MLRI_difcriB1 + MLRI_difcriR1
        MLRI_difcriR2 = MLRI_difcriR2 + MLRI_difcriB2
        MLRI_difcriB2 = MLRI_difcriB2 + MLRI_difcriR2

        (
            RI_criR1,
            MLRI_criR1,
            RI_criB1,
            MLRI_criB1,
            RI_difcriR1,
            MLRI_difcriR1,
            RI_difcriB1,
            MLRI_difcriB1,
            RI_criR2,
            MLRI_criR2,
            RI_criB2,
            MLRI_criB2,
            RI_difcriR2,
            MLRI_difcriR2,
            RI_difcriB2,
            MLRI_difcriB2,
        ) = _imfilter_many(
            (
                RI_criR1,
                MLRI_criR1,
                RI_criB1,
                MLRI_criB1,
                RI_difcriR1,
                MLRI_difcriR1,
                RI_difcriB1,
                MLRI_difcriB1,
                RI_criR2,
                MLRI_criR2,
                RI_criB2,
                MLRI_criB2,
                RI_difcriR2,
                MLRI_difcriR2,
                RI_difcriB2,
                MLRI_difcriB2,
            ),
            Fg,
            "replicate",
        )
        RI_criR1 = RI_criR1 / M_smooth * imask_g
        MLRI_criR1 = MLRI_criR1 / M_smooth * imask_g
        RI_criB1 = RI_criB1 / M_smooth * imask_g
        MLRI_criB1 = MLRI_criB1 / M_smooth * imask_g
        RI_difcriR1 = RI_difcriR1 / M_smooth * imask_g
        MLRI_difcriR1 = MLRI_difcriR1 / M_smooth * imask_g
        RI_difcriB1 = RI_difcriB1 / M_smooth * imask_g
        MLRI_difcriB1 = MLRI_difcriB1 / M_smooth * imask_g
        RI_criR2 = RI_criR2 / M_smooth * imask_g
        MLRI_criR2 = MLRI_criR2 / M_smooth * imask_g
        RI_criB2 = RI_criB2 / M_smooth * imask_g
        MLRI_criB2 = MLRI_criB2 / M_smooth * imask_g
        RI_difcriR2 = RI_difcriR2 / M_smooth * imask_g
        MLRI_difcriR2 = MLRI_difcriR2 / M_smooth * imask_g
        RI_difcriB2 = RI_difcriB2 / M_smooth * imask_g
        MLRI_difcriB2 = MLRI_difcriB2 / M_smooth * imask_g

        RI_wR1 = RI_criR1**2 * RI_difcriR1
        RI_wR2 = RI_criR2**2 * RI_difcriR2
        MLRI_wR1 = MLRI_criR1**2 * MLRI_difcriR1
        MLRI_wR2 = MLRI_criR2**2 * MLRI_difcriR2
        RI_wB1 = RI_criB1**2 * RI_difcriB1
        RI_wB2 = RI_criB2**2 * RI_difcriB2
        MLRI_wB1 = MLRI_criB1**2 * MLRI_difcriB1
        MLRI_wB2 = MLRI_criB2**2 * MLRI_difcriB2

        RI_piR1 = RI_wR1 < RI_w2R1
        RI_piR2 = RI_wR2 < RI_w2R2
        MLRI_piR1 = MLRI_wR1 < MLRI_w2R1
        MLRI_piR2 = MLRI_wR2 < MLRI_w2R2
        RI_piB1 = RI_wB1 < RI_w2B1
        RI_piB2 = RI_wB2 < RI_w2B2
        MLRI_piB1 = MLRI_wB1 < MLRI_w2B1
        MLRI_piB2 = MLRI_wB2 < MLRI_w2B2

        RI_Guider1 = mosaic_r + RI_R1
        RI_Guideb1 = mosaic_b + RI_B1
        RI_Guider2 = mosaic_r + RI_R2
        RI_Guideb2 = mosaic_b + RI_B2
        MLRI_Guider1 = mosaic_r + MLRI_R1
        MLRI_Guideb1 = mosaic_b + MLRI_B1
        MLRI_Guider2 = mosaic_r + MLRI_R2
        MLRI_Guideb2 = mosaic_b + MLRI_B2

        RI_R1[RI_piR1] = RI_Guider1[RI_piR1]
        MLRI_R1[MLRI_piR1] = MLRI_Guider1[MLRI_piR1]
        RI_R2[RI_piR2] = RI_Guider2[RI_piR2]
        MLRI_R2[MLRI_piR2] = MLRI_Guider2[MLRI_piR2]
        RI_B1[RI_piB1] = RI_Guideb1[RI_piB1]
        MLRI_B1[MLRI_piB1] = MLRI_Guideb1[MLRI_piB1]
        RI_B2[RI_piB2] = RI_Guideb2[RI_piB2]
        MLRI_B2[MLRI_piB2] = MLRI_Guideb2[MLRI_piB2]

        RI_w2R1[RI_piR1] = RI_wR1[RI_piR1]
        RI_w2R2[RI_piR2] = RI_wR2[RI_piR2]
        RI_w2B1[RI_piB1] = RI_wB1[RI_piB1]
        RI_w2B2[RI_piB2] = RI_wB2[RI_piB2]
        MLRI_w2R1[MLRI_piR1] = MLRI_wR1[MLRI_piR1]
        MLRI_w2R2[MLRI_piR2] = MLRI_wR2[MLRI_piR2]
        MLRI_w2B1[MLRI_piB1] = MLRI_wB1[MLRI_piB1]
        MLRI_w2B2[MLRI_piB2] = MLRI_wB2[MLRI_piB2]

        h += 1
        v += 1
        h2 += 1
        v2 += 1

    RI_w2R1 = 1 / (RI_w2R1 + 1e-10)
    RI_w2R2 = 1 / (RI_w2R2 + 1e-10)
    MLRI_w2R1 = 1 / (MLRI_w2R1 + 1e-10)
    MLRI_w2R2 = 1 / (MLRI_w2R2 + 1e-10)
    RI_w2B1 = 1 / (RI_w2B1 + 1e-10)
    RI_w2B2 = 1 / (RI_w2B2 + 1e-10)
    MLRI_w2B1 = 1 / (MLRI_w2B1 + 1e-10)
    MLRI_w2B2 = 1 / (MLRI_w2B2 + 1e-10)

    wR = RI_w2R1 + RI_w2R2 + MLRI_w2R1 + MLRI_w2R2
    wB = RI_w2B1 + RI_w2B2 + MLRI_w2B1 + MLRI_w2B2
    red = (RI_w2R1 * RI_R1 + RI_w2R2 * RI_R2 + MLRI_w2R1 * MLRI_R1 + MLRI_w2R2 * MLRI_R2) / (wR + 1e-32)
    blue = (RI_w2B1 * RI_B1 + RI_w2B2 * RI_B2 + MLRI_w2B1 * MLRI_B1 + MLRI_w2B2 * MLRI_B2) / (wB + 1e-32)

    red = red * mask_b + mosaic_r
    blue = blue * mask_r + mosaic_b
    return _clip(red, 0, 255), _clip(blue, 0, 255)


def _red_blue_interpolation_second(
    green: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    mask: np.ndarray,
    pattern: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask_g = mask[:, :, 1]
    imask_g = ~mask_g

    F1 = _ARI_HALF_H
    F1_red, F1_blue = _imfilter_many((red, blue), F1, "replicate")
    Guider1 = red + F1_red * mask_g
    Guideg1 = green
    Guideb1 = blue + F1_blue * mask_g
    F2 = _ARI_HALF_V
    F2_red, F2_blue = _imfilter_many((red, blue), F2, "replicate")
    Guider2 = red + F2_red * mask_g
    Guideg2 = green
    Guideb2 = blue + F2_blue * mask_g

    h = 2
    v = 2
    h2 = 2
    v2 = 0
    itnum = 2
    base = np.ones(mask_g.shape, dtype=bool)

    RI_w2R1 = np.full(mask_g.shape, 1e32, dtype=np.float64)
    RI_w2R2 = np.full(mask_g.shape, 1e32, dtype=np.float64)
    MLRI_w2R1 = np.full(mask_g.shape, 1e32, dtype=np.float64)
    MLRI_w2R2 = np.full(mask_g.shape, 1e32, dtype=np.float64)
    RI_w2B1 = np.full(mask_g.shape, 1e32, dtype=np.float64)
    RI_w2B2 = np.full(mask_g.shape, 1e32, dtype=np.float64)
    MLRI_w2B1 = np.full(mask_g.shape, 1e32, dtype=np.float64)
    MLRI_w2B2 = np.full(mask_g.shape, 1e32, dtype=np.float64)

    RI_Guideg1 = Guideg1.copy()
    RI_Guider1 = Guider1.copy()
    RI_Guideb1 = Guideb1.copy()
    RI_Guideg2 = Guideg2.copy()
    RI_Guider2 = Guider2.copy()
    RI_Guideb2 = Guideb2.copy()
    MLRI_Guideg1 = Guideg1.copy()
    MLRI_Guider1 = Guider1.copy()
    MLRI_Guideb1 = Guideb1.copy()
    MLRI_Guideg2 = Guideg2.copy()
    MLRI_Guider2 = Guider2.copy()
    MLRI_Guideb2 = Guideb2.copy()

    RI_R1 = Guider1.copy()
    RI_R2 = Guider2.copy()
    MLRI_R1 = Guider1.copy()
    MLRI_R2 = Guider2.copy()
    RI_B1 = Guideb1.copy()
    RI_B2 = Guideb2.copy()
    MLRI_B1 = Guideb1.copy()
    MLRI_B2 = Guideb2.copy()

    M = base
    F1_mlri = _ARI_DETAIL_H
    F2_mlri = _ARI_DETAIL_V
    K1_residual = _ARI_HALF_H
    K2_residual = _ARI_HALF_V
    F1_cri = _ARI_CRI_H
    F2_cri = _ARI_CRI_V
    Fg = _gaussian_kernel((5, 5), 2)
    for _ in range(itnum):
        N_M_hv = _boxfilter_ones_count(M.shape, h, v)
        N_M_vh = N_M_hv if h == v else _boxfilter_ones_count(M.shape, v, h)
        RI_tentativeR1, RI_tentativeB1 = _guidedfilter_same_guide(
            RI_Guideg1, RI_Guider1, RI_Guideb1, M, h, v, eps, N=N_M_hv
        )
        RI_tentativeR2, RI_tentativeB2 = _guidedfilter_same_guide(
            RI_Guideg2, RI_Guider2, RI_Guideb2, M, v, h, eps, N=N_M_vh
        )

        difR, difG, difB = _imfilter_many((MLRI_Guider1, MLRI_Guideg1, MLRI_Guideb1), F1_mlri, "replicate")
        N_imask_hv = _bayer_pair_count(mask_g.shape, pattern, "r", "b", h2, v2)
        N3_M_hv = _boxfilter_ones_count(M.shape, h2, v2)
        MLRI_tentativeR1, MLRI_tentativeB1 = _guidedfilter_mlri_same_guide(
            MLRI_Guideg1, MLRI_Guider1, MLRI_Guideb1, M, difG, difR, difB, imask_g, h2, v2, eps, N_imask_hv, N3_M_hv
        )
        difR, difG, difB = _imfilter_many((MLRI_Guider2, MLRI_Guideg2, MLRI_Guideb2), F2_mlri, "replicate")
        N_imask_vh = _bayer_pair_count(mask_g.shape, pattern, "r", "b", v2, h2)
        N3_M_vh = _boxfilter_ones_count(M.shape, v2, h2)
        MLRI_tentativeR2, MLRI_tentativeB2 = _guidedfilter_mlri_same_guide(
            MLRI_Guideg2, MLRI_Guider2, MLRI_Guideb2, M, difG, difR, difB, imask_g, v2, h2, eps, N_imask_vh, N3_M_vh
        )

        RI_residualR1 = (red - RI_tentativeR1) * imask_g
        RI_residualB1 = (blue - RI_tentativeB1) * imask_g
        RI_residualR2 = (red - RI_tentativeR2) * imask_g
        RI_residualB2 = (blue - RI_tentativeB2) * imask_g
        MLRI_residualR1 = (red - MLRI_tentativeR1) * imask_g
        MLRI_residualB1 = (blue - MLRI_tentativeB1) * imask_g
        MLRI_residualR2 = (red - MLRI_tentativeR2) * imask_g
        MLRI_residualB2 = (blue - MLRI_tentativeB2) * imask_g

        RI_residualR1, RI_residualB1, MLRI_residualR1, MLRI_residualB1 = _imfilter_many(
            (RI_residualR1, RI_residualB1, MLRI_residualR1, MLRI_residualB1), K1_residual, "replicate"
        )
        RI_residualR2, RI_residualB2, MLRI_residualR2, MLRI_residualB2 = _imfilter_many(
            (RI_residualR2, RI_residualB2, MLRI_residualR2, MLRI_residualB2), K2_residual, "replicate"
        )

        RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask_g
        RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask_g
        RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask_g
        RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask_g
        MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask_g
        MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask_g
        MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask_g
        MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask_g

        RI_criR1 = (RI_Guider1 - RI_tentativeR1) * M
        RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * M
        RI_criR2 = (RI_Guider2 - RI_tentativeR2) * M
        RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * M
        MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * M
        MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * M
        MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * M
        MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * M

        RI_difcriR1, RI_difcriB1, MLRI_difcriR1, MLRI_difcriB1 = _imfilter_many(
            (RI_criR1, RI_criB1, MLRI_criR1, MLRI_criB1), F1_cri, "replicate"
        )
        RI_difcriR1 = np.abs(RI_difcriR1)
        RI_difcriB1 = np.abs(RI_difcriB1)
        MLRI_difcriR1 = np.abs(MLRI_difcriR1)
        MLRI_difcriB1 = np.abs(MLRI_difcriB1)
        RI_difcriR2, RI_difcriB2, MLRI_difcriR2, MLRI_difcriB2 = _imfilter_many(
            (RI_criR2, RI_criB2, MLRI_criR2, MLRI_criB2), F2_cri, "replicate"
        )
        RI_difcriR2 = np.abs(RI_difcriR2)
        RI_difcriB2 = np.abs(RI_difcriB2)
        MLRI_difcriR2 = np.abs(MLRI_difcriR2)
        MLRI_difcriB2 = np.abs(MLRI_difcriB2)

        RI_criR1 = np.abs(RI_criR1)
        RI_criB1 = np.abs(RI_criB1)
        RI_criR2 = np.abs(RI_criR2)
        RI_criB2 = np.abs(RI_criB2)
        MLRI_criR1 = np.abs(MLRI_criR1)
        MLRI_criB1 = np.abs(MLRI_criB1)
        MLRI_criR2 = np.abs(MLRI_criR2)
        MLRI_criB2 = np.abs(MLRI_criB2)

        RI_criR1 = RI_criR1 + RI_criB1
        RI_criB1 = RI_criB1 + RI_criR1
        RI_criR2 = RI_criR2 + RI_criB2
        RI_criB2 = RI_criB2 + RI_criR2
        MLRI_criR1 = MLRI_criR1 + MLRI_criB1
        MLRI_criB1 = MLRI_criB1 + MLRI_criR1
        MLRI_criR2 = MLRI_criR2 + MLRI_criB2
        MLRI_criB2 = MLRI_criB2 + MLRI_criR2

        RI_difcriR1 = RI_difcriR1 + RI_difcriB1
        RI_difcriB1 = RI_difcriB1 + RI_difcriR1
        RI_difcriR2 = RI_difcriR2 + RI_difcriB2
        RI_difcriB2 = RI_difcriB2 + RI_difcriR2
        MLRI_difcriR1 = MLRI_difcriR1 + MLRI_difcriB1
        MLRI_difcriB1 = MLRI_difcriB1 + MLRI_difcriR1
        MLRI_difcriR2 = MLRI_difcriR2 + MLRI_difcriB2
        MLRI_difcriB2 = MLRI_difcriB2 + MLRI_difcriR2

        (
            RI_criR1,
            MLRI_criR1,
            RI_criB1,
            MLRI_criB1,
            RI_difcriR1,
            MLRI_difcriR1,
            RI_difcriB1,
            MLRI_difcriB1,
            RI_criR2,
            MLRI_criR2,
            RI_criB2,
            MLRI_criB2,
            RI_difcriR2,
            MLRI_difcriR2,
            RI_difcriB2,
            MLRI_difcriB2,
        ) = _imfilter_many(
            (
                RI_criR1,
                MLRI_criR1,
                RI_criB1,
                MLRI_criB1,
                RI_difcriR1,
                MLRI_difcriR1,
                RI_difcriB1,
                MLRI_difcriB1,
                RI_criR2,
                MLRI_criR2,
                RI_criB2,
                MLRI_criB2,
                RI_difcriR2,
                MLRI_difcriR2,
                RI_difcriB2,
                MLRI_difcriB2,
            ),
            Fg,
            "replicate",
        )

        RI_wR1 = RI_criR1**2 * RI_difcriR1
        RI_wR2 = RI_criR2**2 * RI_difcriR2
        MLRI_wR1 = MLRI_criR1**2 * MLRI_difcriR1
        MLRI_wR2 = MLRI_criR2**2 * MLRI_difcriR2
        RI_wB1 = RI_criB1**2 * RI_difcriB1
        RI_wB2 = RI_criB2**2 * RI_difcriB2
        MLRI_wB1 = MLRI_criB1**2 * MLRI_difcriB1
        MLRI_wB2 = MLRI_criB2**2 * MLRI_difcriB2

        RI_piR1 = RI_wR1 < RI_w2R1
        RI_piR2 = RI_wR2 < RI_w2R2
        MLRI_piR1 = MLRI_wR1 < MLRI_w2R1
        MLRI_piR2 = MLRI_wR2 < MLRI_w2R2
        RI_piB1 = RI_wB1 < RI_w2B1
        RI_piB2 = RI_wB2 < RI_w2B2
        MLRI_piB1 = MLRI_wB1 < MLRI_w2B1
        MLRI_piB2 = MLRI_wB2 < MLRI_w2B2

        RI_Guider1 = red + RI_R1
        RI_Guideb1 = blue + RI_B1
        RI_Guider2 = red + RI_R2
        RI_Guideb2 = blue + RI_B2
        MLRI_Guider1 = red + MLRI_R1
        MLRI_Guideb1 = blue + MLRI_B1
        MLRI_Guider2 = red + MLRI_R2
        MLRI_Guideb2 = blue + MLRI_B2

        RI_R1[RI_piR1] = RI_Guider1[RI_piR1]
        MLRI_R1[MLRI_piR1] = MLRI_Guider1[MLRI_piR1]
        RI_R2[RI_piR2] = RI_Guider2[RI_piR2]
        MLRI_R2[MLRI_piR2] = MLRI_Guider2[MLRI_piR2]
        RI_B1[RI_piB1] = RI_Guideb1[RI_piB1]
        MLRI_B1[MLRI_piB1] = MLRI_Guideb1[MLRI_piB1]
        RI_B2[RI_piB2] = RI_Guideb2[RI_piB2]
        MLRI_B2[MLRI_piB2] = MLRI_Guideb2[MLRI_piB2]

        RI_w2R1[RI_piR1] = RI_wR1[RI_piR1]
        RI_w2R2[RI_piR2] = RI_wR2[RI_piR2]
        RI_w2B1[RI_piB1] = RI_wB1[RI_piB1]
        RI_w2B2[RI_piB2] = RI_wB2[RI_piB2]
        MLRI_w2R1[MLRI_piR1] = MLRI_wR1[MLRI_piR1]
        MLRI_w2R2[MLRI_piR2] = MLRI_wR2[MLRI_piR2]
        MLRI_w2B1[MLRI_piB1] = MLRI_wB1[MLRI_piB1]
        MLRI_w2B2[MLRI_piB2] = MLRI_wB2[MLRI_piB2]

        h += 1
        v += 1
        h2 += 1
        v2 += 1

    RI_w2R1 = 1 / (RI_w2R1 + 1e-10)
    RI_w2R2 = 1 / (RI_w2R2 + 1e-10)
    MLRI_w2R1 = 1 / (MLRI_w2R1 + 1e-10)
    MLRI_w2R2 = 1 / (MLRI_w2R2 + 1e-10)
    RI_w2B1 = 1 / (RI_w2B1 + 1e-10)
    RI_w2B2 = 1 / (RI_w2B2 + 1e-10)
    MLRI_w2B1 = 1 / (MLRI_w2B1 + 1e-10)
    MLRI_w2B2 = 1 / (MLRI_w2B2 + 1e-10)

    wR = RI_w2R1 + RI_w2R2 + MLRI_w2R1 + MLRI_w2R2
    wB = RI_w2B1 + RI_w2B2 + MLRI_w2B1 + MLRI_w2B2
    red2 = (RI_w2R1 * RI_R1 + RI_w2R2 * RI_R2 + MLRI_w2R1 * MLRI_R1 + MLRI_w2R2 * MLRI_R2) / (wR + 1e-32)
    blue2 = (RI_w2B1 * RI_B1 + RI_w2B2 * RI_B2 + MLRI_w2B1 * MLRI_B1 + MLRI_w2B2 * MLRI_B2) / (wB + 1e-32)

    red = red + red2 * mask_g
    blue = blue + blue2 * mask_g
    return _clip(red, 0, 255), _clip(blue, 0, 255)
