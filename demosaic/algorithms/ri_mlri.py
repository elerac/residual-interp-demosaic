from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from demosaic.bayer import mask_gr_gb
from demosaic.matlab_compat import boxfilter, clip, gaussian_kernel

_HALF_H = np.array([[0.5, 0.0, 0.5]])
_HALF_V = _HALF_H.T
_DIFF_H = np.array([[1.0, 0.0, -1.0]])
_DIFF_V = _DIFF_H.T
_RI_WEIGHT_SUM = np.ones((5, 5))
_RI_WEST = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
_RI_EAST = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
_MLRI_WEIGHT_SUM = np.ones((3, 3))
_MLRI_WEST = np.array([[1.0, 0.0, 0.0]])
_MLRI_EAST = np.array([[0.0, 0.0, 1.0]])
_MLRI_DETAIL_H = np.array([[-1.0, 0.0, 2.0, 0.0, -1.0]])
_MLRI_DETAIL_V = _MLRI_DETAIL_H.T
_MLRI_LAPLACIAN = np.array(
    [
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [-1, 0, 4, 0, -1],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
    ],
    dtype=np.float64,
)
_RESIDUAL_KERNEL = np.array(
    [
        [0.25, 0.5, 0.25],
        [0.5, 1.0, 0.5],
        [0.25, 0.5, 0.25],
    ]
)
_BAYER_PHASES = {
    "rggb": {"r": (0, 0), "gr": (0, 1), "gb": (1, 0), "b": (1, 1)},
    "grbg": {"gr": (0, 0), "r": (0, 1), "b": (1, 0), "gb": (1, 1)},
    "gbrg": {"gb": (0, 0), "b": (0, 1), "r": (1, 0), "gr": (1, 1)},
    "bggr": {"b": (0, 0), "gb": (0, 1), "gr": (1, 0), "r": (1, 1)},
}


def imfilter(src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim == 1:
        kernel = kernel.reshape(1, -1)
    if kernel.ndim != 2:
        raise ValueError("kernel must be 1D or 2D")
    if src.ndim == 3:
        return cv2.filter2D(src, cv2.CV_64F, kernel, borderType=cv2.BORDER_REPLICATE)
    return cv2.filter2D(src, cv2.CV_64F, kernel, borderType=cv2.BORDER_REPLICATE)


def _imfilter_many(srcs: tuple[np.ndarray, ...], kernel: np.ndarray) -> tuple[np.ndarray, ...]:
    if len(srcs) == 1:
        return (imfilter(srcs[0], kernel),)
    out = imfilter(np.stack([np.asarray(src, dtype=np.float64) for src in srcs], axis=2), kernel)
    return tuple(out[:, :, index] for index in range(len(srcs)))


def _replicate_shift(src: np.ndarray, row_offset: int, col_offset: int) -> np.ndarray:
    rows, cols = _replicate_shift_indices(src.shape, row_offset, col_offset)
    if row_offset == 0:
        return src[:, cols]
    if col_offset == 0:
        return src[rows, :]
    return src[rows[:, None], cols[None, :]]


@lru_cache(maxsize=None)
def _replicate_shift_indices(shape: tuple[int, int], row_offset: int, col_offset: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.clip(np.arange(shape[0]) + row_offset, 0, shape[0] - 1)
    cols = np.clip(np.arange(shape[1]) + col_offset, 0, shape[1] - 1)
    rows.setflags(write=False)
    cols.setflags(write=False)
    return rows, cols


def _boxfilter_count(m: np.ndarray, h: int, v: int) -> np.ndarray:
    n = boxfilter(m, h, v)
    n[n == 0] = 1
    return n


def _as_float_mosaic_bool_mask(mosaic: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mosaic = np.asarray(mosaic, dtype=np.float64)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        mask = mask != 0
    if mosaic.ndim != 3 or mask.ndim != 3 or mosaic.shape != mask.shape or mosaic.shape[2] != 3:
        raise ValueError("mosaic and mask must both have shape (height, width, 3)")
    return mosaic, mask


def _bayer_count(shape: tuple[int, int], pattern: str, key: str, h: int, v: int) -> np.ndarray:
    row_phase, col_phase = _BAYER_PHASES[pattern.lower()][key]
    return _bayer_phase_count(shape, row_phase, col_phase, h, v)


@lru_cache(maxsize=None)
def _bayer_phase_count(shape: tuple[int, int], row_phase: int, col_phase: int, h: int, v: int) -> np.ndarray:
    rows = _parity_window_counts(shape[0], v, row_phase)
    cols = _parity_window_counts(shape[1], h, col_phase)
    n = rows[:, None].astype(np.float64) * cols[None, :].astype(np.float64)
    n[n == 0] = 1
    n.setflags(write=False)
    return n


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


@lru_cache(maxsize=None)
def _boxfilter_ones(shape: tuple[int, int], h: int, v: int) -> np.ndarray:
    if h == 0 and v == 0:
        n = np.zeros(shape, dtype=np.float64)
    else:
        rows = np.arange(shape[0])
        cols = np.arange(shape[1])
        row_lengths = np.minimum(rows + v + 1, shape[0]) - np.maximum(rows - v, 0)
        col_lengths = np.minimum(cols + h + 1, shape[1]) - np.maximum(cols - h, 0)
        n = row_lengths[:, None].astype(np.float64) * col_lengths[None, :].astype(np.float64)
    n.setflags(write=False)
    return n


def _boxfilter_many(srcs: tuple[np.ndarray, ...], h: int, v: int) -> tuple[np.ndarray, ...]:
    return tuple(boxfilter(src, h, v) for src in srcs)


@lru_cache(maxsize=None)
def _directional_gaussian_kernels(sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hwin = gaussian_kernel((1, 9), sigma)
    east = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=np.float64) * hwin
    west = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.float64) * hwin
    east = east / np.sum(east, axis=1, keepdims=True)
    west = west / np.sum(west, axis=1, keepdims=True)
    return west, east, west.T, east.T


def _guidedfilter_ri(
    i: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    h: int,
    v: int,
    eps: float,
    n: np.ndarray | None = None,
) -> np.ndarray:
    th = 0.00001 * 255 * 255
    if n is None:
        n = _boxfilter_count(m, h, v)
    n2 = _boxfilter_ones(i.shape, h, v)

    mean_i = boxfilter(i * m, h, v) / n
    mean_p = boxfilter(p, h, v) / n
    mean_ip = boxfilter(i * p, h, v) / n
    cov_ip = mean_ip - mean_i * mean_p
    mean_ii = boxfilter(i * i * m, h, v) / n
    var_i = mean_ii - mean_i * mean_i
    var_i[var_i < th] = th

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = boxfilter(a, h, v) / n2
    mean_b = boxfilter(b, h, v) / n2
    return mean_a * i + mean_b


def _guidedfilter_mlri(
    g: np.ndarray,
    r: np.ndarray,
    mask: np.ndarray,
    i: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    h: int,
    v: int,
    eps: float,
    n: np.ndarray | None = None,
) -> np.ndarray:
    th = 0.00001 * 255 * 255
    if n is None:
        n = _boxfilter_count(m, h, v)
    n2 = _boxfilter_ones(i.shape, h, v)

    mean_ip = boxfilter(i * p * m, h, v) / n
    mean_ii = boxfilter(i * i * m, h, v) / n
    mean_ii[mean_ii < th] = th
    a = mean_ip / (mean_ii + eps)

    n3 = n if mask is m else _boxfilter_count(mask, h, v)
    mean_g = boxfilter(g * mask, h, v) / n3
    mean_r = boxfilter(r * mask, h, v) / n3
    b = mean_r - a * mean_g

    mean_a = boxfilter(a, h, v) / n2
    mean_b = boxfilter(b, h, v) / n2
    return mean_a * g + mean_b


def _guidedfilter_mlri_weighted(
    g: np.ndarray,
    r: np.ndarray,
    mask: np.ndarray,
    i: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    h: int,
    v: int,
    eps: float,
    n: np.ndarray | None = None,
) -> np.ndarray:
    if n is None:
        n = _boxfilter_count(m, h, v)

    sum_ip, sum_ii, sum_g, sum_r, sum_gg, sum_rr, sum_rg = _boxfilter_many(
        (i * p * m, i * i * m, g * mask, r * mask, g * g * mask, r * r * mask, r * g * mask), h, v
    )
    mean_ip = sum_ip / n
    mean_ii = sum_ii / n
    a = mean_ip / (mean_ii + eps)

    n3 = n if mask is m else _boxfilter_count(mask, h, v)
    mean_g = sum_g / n3
    mean_r = sum_r / n3
    b = mean_r - a * mean_g

    dif = (
        sum_gg * a * a
        + b * b * n3
        + sum_rr
        + 2 * a * b * sum_g
        - 2 * b * sum_r
        - 2 * a * sum_rg
    )
    dif = dif / n3
    dif[dif < 0.01] = 0.01
    dif = 1.0 / dif
    wdif = boxfilter(dif, h, v)
    wdif[wdif < 0.01] = 0.01
    mean_a = boxfilter(a * dif, h, v) / wdif
    mean_b = boxfilter(b * dif, h, v) / wdif
    return mean_a * g + mean_b


def _ri_green_interpolation(mosaic: np.ndarray, mask: np.ndarray, pattern: str, sigma: float) -> np.ndarray:
    mosaic_r = mosaic[:, :, 0]
    mosaic_g = mosaic[:, :, 1]
    mosaic_b = mosaic[:, :, 2]
    rawq = mosaic_r + mosaic_g + mosaic_b
    mask_gr, mask_gb = mask_gr_gb(rawq.shape, pattern)
    mask_r = mask[:, :, 0]
    mask_g = mask[:, :, 1]
    mask_b = mask[:, :, 2]
    imask_g = ~mask_g
    mosaic_g_gr = mosaic_g * mask_gr
    mosaic_g_gb = mosaic_g * mask_gb

    rawh = imfilter(rawq, _HALF_H)
    rawv = imfilter(rawq, _HALF_V)

    guide_gh = mosaic_g + rawh * imask_g
    guide_rh = mosaic_r + rawh * mask_gr
    guide_bh = mosaic_b + rawh * mask_gb
    guide_gv = mosaic_g + rawv * imask_g
    guide_rv = mosaic_r + rawv * mask_gb
    guide_bv = mosaic_b + rawv * mask_gr

    h, v, eps = 5, 0, 0.0
    n_r_h = _bayer_count(rawq.shape, pattern, "r", h, v)
    n_gr_h = _bayer_count(rawq.shape, pattern, "gr", h, v)
    n_gb_h = _bayer_count(rawq.shape, pattern, "gb", h, v)
    n_b_h = _bayer_count(rawq.shape, pattern, "b", h, v)
    n_r_v = _bayer_count(rawq.shape, pattern, "r", v, h)
    n_gb_v = _bayer_count(rawq.shape, pattern, "gb", v, h)
    n_gr_v = _bayer_count(rawq.shape, pattern, "gr", v, h)
    n_b_v = _bayer_count(rawq.shape, pattern, "b", v, h)
    tentative_rh = _guidedfilter_ri(guide_gh, mosaic_r, mask_r, h, v, eps, n=n_r_h)
    tentative_grh = _guidedfilter_ri(guide_rh, mosaic_g_gr, mask_gr, h, v, eps, n=n_gr_h)
    tentative_gbh = _guidedfilter_ri(guide_bh, mosaic_g_gb, mask_gb, h, v, eps, n=n_gb_h)
    tentative_bh = _guidedfilter_ri(guide_gh, mosaic_b, mask_b, h, v, eps, n=n_b_h)
    tentative_rv = _guidedfilter_ri(guide_gv, mosaic_r, mask_r, v, h, eps, n=n_r_v)
    tentative_grv = _guidedfilter_ri(guide_rv, mosaic_g_gb, mask_gb, v, h, eps, n=n_gb_v)
    tentative_gbv = _guidedfilter_ri(guide_bv, mosaic_g_gr, mask_gr, v, h, eps, n=n_gr_v)
    tentative_bv = _guidedfilter_ri(guide_gv, mosaic_b, mask_b, v, h, eps, n=n_b_v)

    residual_grh = (mosaic_g - tentative_grh) * mask_gr
    residual_gbh = (mosaic_g - tentative_gbh) * mask_gb
    residual_rh = (mosaic_r - tentative_rh) * mask_r
    residual_bh = (mosaic_b - tentative_bh) * mask_b
    residual_grv = (mosaic_g - tentative_grv) * mask_gb
    residual_gbv = (mosaic_g - tentative_gbv) * mask_gr
    residual_rv = (mosaic_r - tentative_rv) * mask_r
    residual_bv = (mosaic_b - tentative_bv) * mask_b

    residual_grh, residual_gbh, residual_rh, residual_bh = _imfilter_many(
        (residual_grh, residual_gbh, residual_rh, residual_bh), _HALF_H
    )
    residual_grv, residual_gbv, residual_rv, residual_bv = _imfilter_many(
        (residual_grv, residual_gbv, residual_rv, residual_bv), _HALF_V
    )

    grh = (tentative_grh + residual_grh) * mask_r
    gbh = (tentative_gbh + residual_gbh) * mask_b
    rh = (tentative_rh + residual_rh) * mask_gr
    bh = (tentative_bh + residual_bh) * mask_gb
    grv = (tentative_grv + residual_grv) * mask_r
    gbv = (tentative_gbv + residual_gbv) * mask_b
    rv = (tentative_rv + residual_rv) * mask_gb
    bv = (tentative_bv + residual_bv) * mask_gr

    difh = mosaic_g + grh + gbh - mosaic_r - mosaic_b - rh - bh
    difv = mosaic_g + grv + gbv - mosaic_r - mosaic_b - rv - bv

    difh2 = np.abs(imfilter(difh, _DIFF_H))
    difv2 = np.abs(imfilter(difv, _DIFF_V))
    wh, wv = _imfilter_many((difh2, difv2), _RI_WEIGHT_SUM)
    ww = 1.0 / (_replicate_shift(wh, 0, -2) ** 2 + 1e-32)
    we = 1.0 / (_replicate_shift(wh, 0, 2) ** 2 + 1e-32)
    wn = 1.0 / (_replicate_shift(wv, -2, 0) ** 2 + 1e-32)
    ws = 1.0 / (_replicate_shift(wv, 2, 0) ** 2 + 1e-32)

    kw, ke, kn, ks = _directional_gaussian_kernels(sigma)
    difn = imfilter(difv, kn)
    difs = imfilter(difv, ks)
    difw = imfilter(difh, kw)
    dife = imfilter(difh, ke)
    wt = ww + we + wn + ws
    dif = (wn * difn + ws * difs + ww * difw + we * dife) / wt
    green = dif + rawq
    green = green * imask_g + rawq * mask_g
    return clip(green, 0, 255)


def _mlri_green_interpolation(
    mosaic: np.ndarray,
    mask: np.ndarray,
    pattern: str,
    sigma: float,
    eps: float,
    weighted: bool,
) -> np.ndarray:
    gf = _guidedfilter_mlri_weighted if weighted else _guidedfilter_mlri
    mosaic_r = mosaic[:, :, 0]
    mosaic_g = mosaic[:, :, 1]
    mosaic_b = mosaic[:, :, 2]
    rawq = mosaic_r + mosaic_g + mosaic_b
    mask_gr, mask_gb = mask_gr_gb(rawq.shape, pattern)
    mask_r = mask[:, :, 0]
    mask_g = mask[:, :, 1]
    mask_b = mask[:, :, 2]
    imask_g = ~mask_g
    mosaic_g_gr = mosaic_g * mask_gr
    mosaic_g_gb = mosaic_g * mask_gb

    rawh = imfilter(rawq, _HALF_H)
    rawv = imfilter(rawq, _HALF_V)
    guide_gh = mosaic_g + rawh * imask_g
    guide_rh = mosaic_r + rawh * mask_gr
    guide_bh = mosaic_b + rawh * mask_gb
    guide_gv = mosaic_g + rawv * imask_g
    guide_rv = mosaic_r + rawv * mask_gb
    guide_bv = mosaic_b + rawv * mask_gr

    h, v = 3, 3
    n_r = _bayer_count(rawq.shape, pattern, "r", h, v)
    n_b = _bayer_count(rawq.shape, pattern, "b", h, v)
    n_gr = _bayer_count(rawq.shape, pattern, "gr", h, v)
    n_gb = _bayer_count(rawq.shape, pattern, "gb", h, v)
    (
        dif_r_h,
        dif_gr_for_r_h,
        dif_green_gr_h,
        dif_r_for_gr_h,
        dif_b_h,
        dif_gb_for_b_h,
        dif_green_gb_h,
        dif_b_for_gb_h,
    ) = _imfilter_many(
        (
            mosaic_r,
            guide_gh * mask_r,
            mosaic_g_gr,
            guide_rh * mask_gr,
            mosaic_b,
            guide_gh * mask_b,
            mosaic_g_gb,
            guide_bh * mask_gb,
        ),
        _MLRI_DETAIL_H,
    )
    tentative_rh = gf(guide_gh, mosaic_r, mask_r, dif_gr_for_r_h, dif_r_h, mask_r, h, v, eps, n=n_r)
    tentative_grh = gf(guide_rh, mosaic_g_gr, mask_gr, dif_r_for_gr_h, dif_green_gr_h, mask_gr, h, v, eps, n=n_gr)
    tentative_bh = gf(guide_gh, mosaic_b, mask_b, dif_gb_for_b_h, dif_b_h, mask_b, h, v, eps, n=n_b)
    tentative_gbh = gf(guide_bh, mosaic_g_gb, mask_gb, dif_b_for_gb_h, dif_green_gb_h, mask_gb, h, v, eps, n=n_gb)

    (
        dif_r_v,
        dif_gr_for_r_v,
        dif_green_gb_v,
        dif_r_for_gb_v,
        dif_b_v,
        dif_gb_for_b_v,
        dif_green_gr_v,
        dif_b_for_gr_v,
    ) = _imfilter_many(
        (
            mosaic_r,
            guide_gv * mask_r,
            mosaic_g_gb,
            guide_rv * mask_gb,
            mosaic_b,
            guide_gv * mask_b,
            mosaic_g_gr,
            guide_bv * mask_gr,
        ),
        _MLRI_DETAIL_V,
    )
    tentative_rv = gf(guide_gv, mosaic_r, mask_r, dif_gr_for_r_v, dif_r_v, mask_r, v, h, eps, n=n_r)
    tentative_grv = gf(guide_rv, mosaic_g_gb, mask_gb, dif_r_for_gb_v, dif_green_gb_v, mask_gb, v, h, eps, n=n_gb)
    tentative_bv = gf(guide_gv, mosaic_b, mask_b, dif_gb_for_b_v, dif_b_v, mask_b, v, h, eps, n=n_b)
    tentative_gbv = gf(guide_bv, mosaic_g_gr, mask_gr, dif_b_for_gr_v, dif_green_gr_v, mask_gr, v, h, eps, n=n_gr)

    tentative_grh = clip(tentative_grh, 0, 255)
    tentative_grv = clip(tentative_grv, 0, 255)
    tentative_gbh = clip(tentative_gbh, 0, 255)
    tentative_gbv = clip(tentative_gbv, 0, 255)
    tentative_rh = clip(tentative_rh, 0, 255)
    tentative_rv = clip(tentative_rv, 0, 255)
    tentative_bh = clip(tentative_bh, 0, 255)
    tentative_bv = clip(tentative_bv, 0, 255)

    residual_grh = (mosaic_g - tentative_grh) * mask_gr
    residual_gbh = (mosaic_g - tentative_gbh) * mask_gb
    residual_rh = (mosaic_r - tentative_rh) * mask_r
    residual_bh = (mosaic_b - tentative_bh) * mask_b
    residual_grv = (mosaic_g - tentative_grv) * mask_gb
    residual_gbv = (mosaic_g - tentative_gbv) * mask_gr
    residual_rv = (mosaic_r - tentative_rv) * mask_r
    residual_bv = (mosaic_b - tentative_bv) * mask_b

    residual_grh, residual_gbh, residual_rh, residual_bh = _imfilter_many(
        (residual_grh, residual_gbh, residual_rh, residual_bh), _HALF_H
    )
    residual_grv, residual_gbv, residual_rv, residual_bv = _imfilter_many(
        (residual_grv, residual_gbv, residual_rv, residual_bv), _HALF_V
    )

    grh = clip((tentative_grh + residual_grh) * mask_r, 0, 255)
    gbh = clip((tentative_gbh + residual_gbh) * mask_b, 0, 255)
    rh = clip((tentative_rh + residual_rh) * mask_gr, 0, 255)
    bh = clip((tentative_bh + residual_bh) * mask_gb, 0, 255)
    grv = clip((tentative_grv + residual_grv) * mask_r, 0, 255)
    gbv = clip((tentative_gbv + residual_gbv) * mask_b, 0, 255)
    rv = clip((tentative_rv + residual_rv) * mask_gb, 0, 255)
    bv = clip((tentative_bv + residual_bv) * mask_gr, 0, 255)

    difh = mosaic_g + grh + gbh - mosaic_r - mosaic_b - rh - bh
    difv = mosaic_g + grv + gbv - mosaic_r - mosaic_b - rv - bv

    difh2 = np.abs(imfilter(difh, _DIFF_H))
    difv2 = np.abs(imfilter(difv, _DIFF_V))
    wh, wv = _imfilter_many((difh2, difv2), _MLRI_WEIGHT_SUM)
    ww = 1.0 / (_replicate_shift(wh, 0, -1) ** 2 + 1e-2)
    we = 1.0 / (_replicate_shift(wh, 0, 1) ** 2 + 1e-2)
    wn = 1.0 / (_replicate_shift(wv, -1, 0) ** 2 + 1e-2)
    ws = 1.0 / (_replicate_shift(wv, 1, 0) ** 2 + 1e-2)

    kw, ke, kn, ks = _directional_gaussian_kernels(sigma)
    difn = imfilter(difv, kn)
    difs = imfilter(difv, ks)
    difw = imfilter(difh, kw)
    dife = imfilter(difh, ke)
    wt = ww + we + wn + ws
    dif = (wn * difn + ws * difs + ww * difw + we * dife) / wt
    green = dif + rawq
    return green * imask_g + rawq * mask_g


def _ri_red_blue(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, channel: int) -> np.ndarray:
    h, v, eps = 5, 5, 0.0
    mosaic_channel = mosaic[:, :, channel]
    mask_channel = mask[:, :, channel]
    tentative = _guidedfilter_ri(green, mosaic_channel, mask_channel, h, v, eps)
    residual = (mosaic_channel - tentative) * mask_channel
    return imfilter(residual, _RESIDUAL_KERNEL) + tentative


def _ri_red_blue_pair(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    h, v, eps = 5, 5, 0.0
    mosaic_r = mosaic[:, :, 0]
    mosaic_b = mosaic[:, :, 2]
    mask_r = mask[:, :, 0]
    mask_b = mask[:, :, 2]
    n_r = _bayer_count(green.shape, pattern, "r", h, v)
    n_b = _bayer_count(green.shape, pattern, "b", h, v)
    tentative_r = _guidedfilter_ri(green, mosaic_r, mask_r, h, v, eps, n=n_r)
    tentative_b = _guidedfilter_ri(green, mosaic_b, mask_b, h, v, eps, n=n_b)
    residual_r = (mosaic_r - tentative_r) * mask_r
    residual_b = (mosaic_b - tentative_b) * mask_b
    residual_r, residual_b = _imfilter_many((residual_r, residual_b), _RESIDUAL_KERNEL)
    return residual_r + tentative_r, residual_b + tentative_b


def _mlri_red_blue(
    green: np.ndarray,
    mosaic: np.ndarray,
    mask: np.ndarray,
    channel: int,
    eps: float,
    weighted: bool,
) -> np.ndarray:
    gf = _guidedfilter_mlri_weighted if weighted else _guidedfilter_mlri
    h, v = 5, 5
    mosaic_channel = mosaic[:, :, channel]
    mask_channel = mask[:, :, channel]
    lap_color = imfilter(mosaic_channel, _MLRI_LAPLACIAN)
    lap_green = imfilter(green * mask_channel, _MLRI_LAPLACIAN)
    tentative = gf(green, mosaic_channel, mask_channel, lap_green, lap_color, mask_channel, h, v, eps)
    tentative = clip(tentative, 0, 255)
    residual = mask_channel * (mosaic_channel - tentative)
    return imfilter(residual, _RESIDUAL_KERNEL) + tentative


def _mlri_red_blue_pair(
    green: np.ndarray,
    mosaic: np.ndarray,
    mask: np.ndarray,
    pattern: str,
    eps: float,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    gf = _guidedfilter_mlri_weighted if weighted else _guidedfilter_mlri
    h, v = 5, 5
    mosaic_r = mosaic[:, :, 0]
    mosaic_b = mosaic[:, :, 2]
    mask_r = mask[:, :, 0]
    mask_b = mask[:, :, 2]
    n_r = _bayer_count(green.shape, pattern, "r", h, v)
    n_b = _bayer_count(green.shape, pattern, "b", h, v)
    lap_r, lap_green_r, lap_b, lap_green_b = _imfilter_many(
        (mosaic_r, green * mask_r, mosaic_b, green * mask_b), _MLRI_LAPLACIAN
    )
    tentative_r = gf(green, mosaic_r, mask_r, lap_green_r, lap_r, mask_r, h, v, eps, n=n_r)
    tentative_r = clip(tentative_r, 0, 255)
    tentative_b = gf(green, mosaic_b, mask_b, lap_green_b, lap_b, mask_b, h, v, eps, n=n_b)
    tentative_b = clip(tentative_b, 0, 255)
    residual_r = mask_r * (mosaic_r - tentative_r)
    residual_b = mask_b * (mosaic_b - tentative_b)
    residual_r, residual_b = _imfilter_many((residual_r, residual_b), _RESIDUAL_KERNEL)
    return residual_r + tentative_r, residual_b + tentative_b


def demosaic_ri(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    mosaic, mask = _as_float_mosaic_bool_mask(mosaic, mask)
    green = _ri_green_interpolation(mosaic, mask, pattern, sigma=1.0)
    red, blue = _ri_red_blue_pair(green, mosaic, mask, pattern)
    out = np.empty((*green.shape, 3), dtype=np.float64)
    out[:, :, 0] = red
    out[:, :, 1] = green
    out[:, :, 2] = blue
    return out


def demosaic_mlri(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    mosaic, mask = _as_float_mosaic_bool_mask(mosaic, mask)
    green = _mlri_green_interpolation(mosaic, mask, pattern, sigma=1.4, eps=0.0, weighted=False)
    green = clip(green, 0, 255)
    red, blue = _mlri_red_blue_pair(green, mosaic, mask, pattern, eps=0.0, weighted=False)
    red = clip(red, 0, 255)
    blue = clip(blue, 0, 255)
    out = np.empty((*green.shape, 3), dtype=np.float64)
    out[:, :, 0] = red
    out[:, :, 1] = green
    out[:, :, 2] = blue
    return out


def demosaic_mlri2(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    mosaic, mask = _as_float_mosaic_bool_mask(mosaic, mask)
    eps = 1e-32
    green = _mlri_green_interpolation(mosaic, mask, pattern, sigma=1.0, eps=eps, weighted=True)
    green = clip(green, 0, 255)
    red, blue = _mlri_red_blue_pair(green, mosaic, mask, pattern, eps=eps, weighted=True)
    red = clip(red, 0, 255)
    blue = clip(blue, 0, 255)
    out = np.empty((*green.shape, 3), dtype=np.float64)
    out[:, :, 0] = red
    out[:, :, 1] = green
    out[:, :, 2] = blue
    return out
