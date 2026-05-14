"""Adaptive residual interpolation demosaicing.

This module translates the MATLAB ARI and ARI2 implementations under
``matlab/algorithms/ARI`` and ``matlab/algorithms/ARI2``.
"""

from __future__ import annotations

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


def demosaic_ari(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    """Demosaic a Bayer image with ARI, returning RGB ``float64``."""
    mosaic, mask = _as_float_inputs(mosaic, mask)
    eps = 1e-10
    h = 5
    v = 5

    green = _green_interpolation(mosaic, mask, pattern, eps)
    red = _red_interpolation(green, mosaic, mask, h, v, eps)
    blue = _blue_interpolation(green, mosaic, mask, h, v, eps)

    rgb_dem = np.zeros_like(mosaic, dtype=np.float64)
    rgb_dem[:, :, 0] = red
    rgb_dem[:, :, 1] = green
    rgb_dem[:, :, 2] = blue
    return rgb_dem


def demosaic_ari2(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    """Demosaic a Bayer image with ARI2, returning RGB ``float64``."""
    mosaic, mask = _as_float_inputs(mosaic, mask)
    eps = 1e-10

    green = _green_interpolation(mosaic, mask, pattern, eps)
    red, blue = _red_blue_interpolation_first(green, mosaic, mask, eps)
    red, blue = _red_blue_interpolation_second(green, red, blue, mask, eps)

    rgb_dem = np.zeros_like(mosaic, dtype=np.float64)
    rgb_dem[:, :, 0] = red
    rgb_dem[:, :, 1] = green
    rgb_dem[:, :, 2] = blue
    return rgb_dem


def _as_float_inputs(mosaic: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mosaic = np.asarray(mosaic, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if mosaic.ndim != 3 or mask.ndim != 3 or mosaic.shape != mask.shape or mosaic.shape[2] != 3:
        raise ValueError("mosaic and mask must both have shape (height, width, 3)")
    return mosaic, mask


def _imfilter(src: np.ndarray, kernel: np.ndarray | list, boundary: str = "replicate") -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    if _shared_imfilter is not None:
        if boundary != "replicate":
            raise ValueError("only replicate boundary is supported")
        return np.asarray(_shared_imfilter(src, kernel), dtype=np.float64)

    from scipy import ndimage

    if boundary != "replicate":
        raise ValueError("only replicate boundary is supported")
    return ndimage.correlate(np.asarray(src, dtype=np.float64), kernel, mode="nearest")


def _boxfilter(src: np.ndarray, h: int, v: int) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    if h == 0 and v == 0:
        return np.zeros_like(src)
    hei, wid = src.shape
    integral = np.pad(src, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    rows = np.arange(hei)
    cols = np.arange(wid)
    r0 = np.maximum(rows - v, 0)
    r1 = np.minimum(rows + v + 1, hei)
    c0 = np.maximum(cols - h, 0)
    c1 = np.minimum(cols + h + 1, wid)
    return (
        integral[r1[:, None], c1[None, :]]
        - integral[r0[:, None], c1[None, :]]
        - integral[r1[:, None], c0[None, :]]
        + integral[r0[:, None], c0[None, :]]
    )


def _mask_gr_gb(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, np.ndarray]:
    if _shared_mask_gr_gb is not None:
        for args in ((shape, pattern), (shape[0], shape[1], pattern), (np.zeros(shape), pattern)):
            try:
                mask_gr, mask_gb = _shared_mask_gr_gb(*args)
                return np.asarray(mask_gr, dtype=np.float64), np.asarray(mask_gb, dtype=np.float64)
            except TypeError:
                continue

    pattern = pattern.lower()
    mask_gr = np.zeros(shape, dtype=np.float64)
    mask_gb = np.zeros(shape, dtype=np.float64)
    if pattern == "grbg":
        mask_gr[0::2, 0::2] = 1
        mask_gb[1::2, 1::2] = 1
    elif pattern == "rggb":
        mask_gr[0::2, 1::2] = 1
        mask_gb[1::2, 0::2] = 1
    elif pattern == "gbrg":
        mask_gb[0::2, 0::2] = 1
        mask_gr[1::2, 1::2] = 1
    elif pattern == "bggr":
        mask_gb[0::2, 1::2] = 1
        mask_gr[1::2, 0::2] = 1
    else:
        raise ValueError(f"unsupported Bayer pattern: {pattern!r}")
    return mask_gr, mask_gb


def _gaussian_kernel(size: tuple[int, int] = (5, 5), sigma: float = 2.0) -> np.ndarray:
    rows, cols = size
    y = np.arange(rows, dtype=np.float64) - (rows - 1) / 2
    x = np.arange(cols, dtype=np.float64) - (cols - 1) / 2
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return kernel / kernel.sum()


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


def _guidedfilter(I: np.ndarray, p: np.ndarray, M: np.ndarray, h: int, v: int, eps: float) -> np.ndarray:
    hei, wid = I.shape
    N = _boxfilter(M, h, v)
    N[N == 0] = 1

    mean_I = _boxfilter(I * M, h, v) / N
    mean_p = _boxfilter(p * M, h, v) / N
    mean_Ip = _boxfilter(I * p * M, h, v) / N

    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = _boxfilter(I * I * M, h, v) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    dif = (
        _boxfilter(I * I * M, h, v) * a * a
        + b * b * N
        + _boxfilter(p * p * M, h, v)
        + 2 * a * b * _boxfilter(I * M, h, v)
        - 2 * b * _boxfilter(p * M, h, v)
        - 2 * a * _boxfilter(p * I * M, h, v)
    )
    dif = np.sqrt(np.maximum(dif / N, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif
    wdif = _boxfilter(dif, h, v)
    mean_a = _boxfilter(a * dif, h, v) / (wdif + 1e-4)
    mean_b = _boxfilter(b * dif, h, v) / (wdif + 1e-4)

    return mean_a * I + mean_b


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
) -> np.ndarray:
    N = _boxfilter(M, h, v)
    N[N == 0] = 1

    mean_Ip = _boxfilter(I * p * M, h, v) / N
    mean_II = _boxfilter(I * I * M, h, v) / N

    a = mean_Ip / (mean_II + eps)
    N3 = _boxfilter(mask, h, v)
    N3[N3 == 0] = 1
    mean_G = _boxfilter(G * mask, h, v) / N3
    mean_R = _boxfilter(R * mask, h, v) / N3
    b = mean_R - a * mean_G

    dif = (
        _boxfilter(G * G * mask, h, v) * a * a
        + b * b * N3
        + _boxfilter(R * R * mask, h, v)
        + 2 * a * b * _boxfilter(G * mask, h, v)
        - 2 * b * _boxfilter(R * mask, h, v)
        - 2 * a * _boxfilter(R * G * mask, h, v)
    )
    dif = np.sqrt(np.maximum(dif / N3, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif
    wdif = _boxfilter(dif, h, v)
    mean_a = _boxfilter(a * dif, h, v) / (wdif + 1e-4)
    mean_b = _boxfilter(b * dif, h, v) / (wdif + 1e-4)

    return mean_a * G + mean_b


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


def _guidedfilter_diagonal(I: np.ndarray, p: np.ndarray, M: np.ndarray, h: int, v: int, eps: float) -> np.ndarray:
    F = _diagonal_window(h, v)

    N = _imfilter(M, F, "replicate")
    N[N == 0] = 1

    mean_I = _imfilter(I * M, F, "replicate") / N
    mean_p = _imfilter(p * M, F, "replicate") / N
    mean_Ip = _imfilter(I * p * M, F, "replicate") / N

    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = _imfilter(I * I * M, F, "replicate") / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    dif = (
        _imfilter(I * I * M, F, "replicate") * a * a
        + b * b * N
        + _imfilter(p * p * M, F, "replicate")
        + 2 * a * b * _imfilter(I * M, F, "replicate")
        - 2 * b * _imfilter(p * M, F, "replicate")
        - 2 * a * _imfilter(p * I * M, F, "replicate")
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
) -> np.ndarray:
    F = _diagonal_window(h, v)

    N = _imfilter(M, F, "replicate")
    N[N == 0] = 1

    mean_Ip = _imfilter(I * p * M, F, "replicate") / N
    mean_II = _imfilter(I * I * M, F, "replicate") / N

    a = mean_Ip / (mean_II + eps)
    N3 = _imfilter(mask, F, "replicate")
    N3[N3 == 0] = 1
    mean_G = _imfilter(G * mask, F, "replicate") / N3
    mean_R = _imfilter(R * mask, F, "replicate") / N3
    b = mean_R - a * mean_G

    dif = (
        _imfilter(G * G * mask, F, "replicate") * a * a
        + b * b * N3
        + _imfilter(R * R * mask, F, "replicate")
        + 2 * a * b * _imfilter(G * mask, F, "replicate")
        - 2 * b * _imfilter(R * mask, F, "replicate")
        - 2 * a * _imfilter(R * G * mask, F, "replicate")
    )
    dif = np.sqrt(np.maximum(dif / N3, 0))
    dif[dif < 1e-3] = 1e-3
    dif = 1 / dif
    wdif = _imfilter(dif, F, "replicate")
    mean_a = _imfilter(a * dif, F, "replicate") / (wdif + 1e-4)
    mean_b = _imfilter(b * dif, F, "replicate") / (wdif + 1e-4)

    return mean_a * G + mean_b


def _green_interpolation(mosaic: np.ndarray, mask: np.ndarray, pattern: str, eps: float) -> np.ndarray:
    imask = (mask == 0).astype(np.float64)
    rawq = mosaic.sum(axis=2)
    mask_gr, mask_gb = _mask_gr_gb(rawq.shape, pattern)

    Mrh = mask[:, :, 0] + mask_gr
    Mbh = mask[:, :, 2] + mask_gb
    Mrv = mask[:, :, 0] + mask_gb
    Mbv = mask[:, :, 2] + mask_gr

    Kh = np.array([[0.5, 0, 0.5]])
    Kv = Kh.T
    rawh = _imfilter(rawq, Kh, "replicate")
    rawv = _imfilter(rawq, Kv, "replicate")

    Guidegrh = mosaic[:, :, 1] * mask_gr + rawh * mask[:, :, 0]
    Guidegbh = mosaic[:, :, 1] * mask_gb + rawh * mask[:, :, 2]
    Guiderh = mosaic[:, :, 0] + rawh * mask_gr
    Guidebh = mosaic[:, :, 2] + rawh * mask_gb
    Guidegrv = mosaic[:, :, 1] * mask_gb + rawv * mask[:, :, 0]
    Guidegbv = mosaic[:, :, 1] * mask_gr + rawv * mask[:, :, 2]
    Guiderv = mosaic[:, :, 0] + rawv * mask_gb
    Guidebv = mosaic[:, :, 2] + rawv * mask_gr

    h = 2
    v = 1
    h2 = 4
    v2 = 0
    itnum = 11

    RI_w2h = np.ones_like(mask_gr) * 1e32
    RI_w2v = np.ones_like(mask_gr) * 1e32
    MLRI_w2h = np.ones_like(mask_gr) * 1e32
    MLRI_w2v = np.ones_like(mask_gr) * 1e32

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

    for _ in range(itnum):
        RI_tentativeGrh = _guidedfilter(RI_Guiderh, RI_Guidegrh, Mrh, h, v, eps)
        RI_tentativeGbh = _guidedfilter(RI_Guidebh, RI_Guidegbh, Mbh, h, v, eps)
        RI_tentativeRh = _guidedfilter(RI_Guidegrh, RI_Guiderh, Mrh, h, v, eps)
        RI_tentativeBh = _guidedfilter(RI_Guidegbh, RI_Guidebh, Mbh, h, v, eps)
        RI_tentativeGrv = _guidedfilter(RI_Guiderv, RI_Guidegrv, Mrv, v, h, eps)
        RI_tentativeGbv = _guidedfilter(RI_Guidebv, RI_Guidegbv, Mbv, v, h, eps)
        RI_tentativeRv = _guidedfilter(RI_Guidegrv, RI_Guiderv, Mrv, v, h, eps)
        RI_tentativeBv = _guidedfilter(RI_Guidegbv, RI_Guidebv, Mbv, v, h, eps)

        Fh = np.array([[-1, 0, 2, 0, -1]], dtype=np.float64)
        difR = _imfilter(MLRI_Guiderh, Fh, "replicate")
        difGr = _imfilter(MLRI_Guidegrh, Fh, "replicate")
        difB = _imfilter(MLRI_Guidebh, Fh, "replicate")
        difGb = _imfilter(MLRI_Guidegbh, Fh, "replicate")
        MLRI_tentativeRh = _guidedfilter_mlri(MLRI_Guidegrh, MLRI_Guiderh, Mrh, difGr, difR, mask[:, :, 0], h2, v2, eps)
        MLRI_tentativeBh = _guidedfilter_mlri(MLRI_Guidegbh, MLRI_Guidebh, Mbh, difGb, difB, mask[:, :, 2], h2, v2, eps)
        MLRI_tentativeGrh = _guidedfilter_mlri(MLRI_Guiderh, MLRI_Guidegrh, Mrh, difR, difGr, mask_gr, h2, v2, eps)
        MLRI_tentativeGbh = _guidedfilter_mlri(MLRI_Guidebh, MLRI_Guidegbh, Mbh, difB, difGb, mask_gb, h2, v2, eps)

        Fv = Fh.T
        difR = _imfilter(MLRI_Guiderv, Fv, "replicate")
        difGr = _imfilter(MLRI_Guidegrv, Fv, "replicate")
        difB = _imfilter(MLRI_Guidebv, Fv, "replicate")
        difGb = _imfilter(MLRI_Guidegbv, Fv, "replicate")
        MLRI_tentativeRv = _guidedfilter_mlri(MLRI_Guidegrv, MLRI_Guiderv, Mrv, difGr, difR, mask[:, :, 0], v2, h2, eps)
        MLRI_tentativeBv = _guidedfilter_mlri(MLRI_Guidegbv, MLRI_Guidebv, Mbv, difGb, difB, mask[:, :, 2], v2, h2, eps)
        MLRI_tentativeGrv = _guidedfilter_mlri(MLRI_Guiderv, MLRI_Guidegrv, Mrv, difR, difGr, mask_gb, v2, h2, eps)
        MLRI_tentativeGbv = _guidedfilter_mlri(MLRI_Guidebv, MLRI_Guidegbv, Mbv, difB, difGb, mask_gr, v2, h2, eps)

        RI_residualGrh = (mosaic[:, :, 1] - RI_tentativeGrh) * mask_gr
        RI_residualGbh = (mosaic[:, :, 1] - RI_tentativeGbh) * mask_gb
        RI_residualRh = (mosaic[:, :, 0] - RI_tentativeRh) * mask[:, :, 0]
        RI_residualBh = (mosaic[:, :, 2] - RI_tentativeBh) * mask[:, :, 2]
        RI_residualGrv = (mosaic[:, :, 1] - RI_tentativeGrv) * mask_gb
        RI_residualGbv = (mosaic[:, :, 1] - RI_tentativeGbv) * mask_gr
        RI_residualRv = (mosaic[:, :, 0] - RI_tentativeRv) * mask[:, :, 0]
        RI_residualBv = (mosaic[:, :, 2] - RI_tentativeBv) * mask[:, :, 2]
        MLRI_residualGrh = (mosaic[:, :, 1] - MLRI_tentativeGrh) * mask_gr
        MLRI_residualGbh = (mosaic[:, :, 1] - MLRI_tentativeGbh) * mask_gb
        MLRI_residualRh = (mosaic[:, :, 0] - MLRI_tentativeRh) * mask[:, :, 0]
        MLRI_residualBh = (mosaic[:, :, 2] - MLRI_tentativeBh) * mask[:, :, 2]
        MLRI_residualGrv = (mosaic[:, :, 1] - MLRI_tentativeGrv) * mask_gb
        MLRI_residualGbv = (mosaic[:, :, 1] - MLRI_tentativeGbv) * mask_gr
        MLRI_residualRv = (mosaic[:, :, 0] - MLRI_tentativeRv) * mask[:, :, 0]
        MLRI_residualBv = (mosaic[:, :, 2] - MLRI_tentativeBv) * mask[:, :, 2]

        Kh = np.array([[0.5, 1, 0.5]], dtype=np.float64)
        RI_residualGrh = _imfilter(RI_residualGrh, Kh, "replicate")
        RI_residualGbh = _imfilter(RI_residualGbh, Kh, "replicate")
        RI_residualRh = _imfilter(RI_residualRh, Kh, "replicate")
        RI_residualBh = _imfilter(RI_residualBh, Kh, "replicate")
        MLRI_residualGrh = _imfilter(MLRI_residualGrh, Kh, "replicate")
        MLRI_residualGbh = _imfilter(MLRI_residualGbh, Kh, "replicate")
        MLRI_residualRh = _imfilter(MLRI_residualRh, Kh, "replicate")
        MLRI_residualBh = _imfilter(MLRI_residualBh, Kh, "replicate")
        Kv = Kh.T
        RI_residualGrv = _imfilter(RI_residualGrv, Kv, "replicate")
        RI_residualGbv = _imfilter(RI_residualGbv, Kv, "replicate")
        RI_residualRv = _imfilter(RI_residualRv, Kv, "replicate")
        RI_residualBv = _imfilter(RI_residualBv, Kv, "replicate")
        MLRI_residualGrv = _imfilter(MLRI_residualGrv, Kv, "replicate")
        MLRI_residualGbv = _imfilter(MLRI_residualGbv, Kv, "replicate")
        MLRI_residualRv = _imfilter(MLRI_residualRv, Kv, "replicate")
        MLRI_residualBv = _imfilter(MLRI_residualBv, Kv, "replicate")

        RI_Grh = (RI_tentativeGrh + RI_residualGrh) * mask[:, :, 0]
        RI_Gbh = (RI_tentativeGbh + RI_residualGbh) * mask[:, :, 2]
        RI_Rh = (RI_tentativeRh + RI_residualRh) * mask_gr
        RI_Bh = (RI_tentativeBh + RI_residualBh) * mask_gb
        RI_Grv = (RI_tentativeGrv + RI_residualGrv) * mask[:, :, 0]
        RI_Gbv = (RI_tentativeGbv + RI_residualGbv) * mask[:, :, 2]
        RI_Rv = (RI_tentativeRv + RI_residualRv) * mask_gb
        RI_Bv = (RI_tentativeBv + RI_residualBv) * mask_gr
        MLRI_Grh = (MLRI_tentativeGrh + MLRI_residualGrh) * mask[:, :, 0]
        MLRI_Gbh = (MLRI_tentativeGbh + MLRI_residualGbh) * mask[:, :, 2]
        MLRI_Rh = (MLRI_tentativeRh + MLRI_residualRh) * mask_gr
        MLRI_Bh = (MLRI_tentativeBh + MLRI_residualBh) * mask_gb
        MLRI_Grv = (MLRI_tentativeGrv + MLRI_residualGrv) * mask[:, :, 0]
        MLRI_Gbv = (MLRI_tentativeGbv + MLRI_residualGbv) * mask[:, :, 2]
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

        Fh = np.array([[-1, 0, 1]], dtype=np.float64)
        RI_difcriGrh = np.abs(_imfilter(RI_criGrh, Fh, "replicate"))
        RI_difcriGbh = np.abs(_imfilter(RI_criGbh, Fh, "replicate"))
        RI_difcriRh = np.abs(_imfilter(RI_criRh, Fh, "replicate"))
        RI_difcriBh = np.abs(_imfilter(RI_criBh, Fh, "replicate"))
        MLRI_difcriGrh = np.abs(_imfilter(MLRI_criGrh, Fh, "replicate"))
        MLRI_difcriGbh = np.abs(_imfilter(MLRI_criGbh, Fh, "replicate"))
        MLRI_difcriRh = np.abs(_imfilter(MLRI_criRh, Fh, "replicate"))
        MLRI_difcriBh = np.abs(_imfilter(MLRI_criBh, Fh, "replicate"))
        Fv = Fh.T
        RI_difcriGrv = np.abs(_imfilter(RI_criGrv, Fv, "replicate"))
        RI_difcriGbv = np.abs(_imfilter(RI_criGbv, Fv, "replicate"))
        RI_difcriRv = np.abs(_imfilter(RI_criRv, Fv, "replicate"))
        RI_difcriBv = np.abs(_imfilter(RI_criBv, Fv, "replicate"))
        MLRI_difcriGrv = np.abs(_imfilter(MLRI_criGrv, Fv, "replicate"))
        MLRI_difcriGbv = np.abs(_imfilter(MLRI_criGbv, Fv, "replicate"))
        MLRI_difcriRv = np.abs(_imfilter(MLRI_criRv, Fv, "replicate"))
        MLRI_difcriBv = np.abs(_imfilter(MLRI_criBv, Fv, "replicate"))

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

        Fg = _gaussian_kernel((5, 5), 2)
        RI_crih = _imfilter(RI_crih, Fg, "replicate")
        MLRI_crih = _imfilter(MLRI_crih, Fg, "replicate")
        RI_difcrih = _imfilter(RI_difcrih, Fg, "replicate")
        MLRI_difcrih = _imfilter(MLRI_difcrih, Fg, "replicate")
        RI_criv = _imfilter(RI_criv, Fg, "replicate")
        MLRI_criv = _imfilter(MLRI_criv, Fg, "replicate")
        RI_difcriv = _imfilter(RI_difcriv, Fg, "replicate")
        MLRI_difcriv = _imfilter(MLRI_difcriv, Fg, "replicate")

        RI_wh = RI_crih**2 * RI_difcrih
        RI_wv = RI_criv**2 * RI_difcriv
        MLRI_wh = MLRI_crih**2 * MLRI_difcrih
        MLRI_wv = MLRI_criv**2 * MLRI_difcriv

        RI_pih = RI_wh < RI_w2h
        RI_piv = RI_wv < RI_w2v
        MLRI_pih = MLRI_wh < MLRI_w2h
        MLRI_piv = MLRI_wv < MLRI_w2v

        RI_Guidegrh = mosaic[:, :, 1] * mask_gr + RI_Grh
        RI_Guidegbh = mosaic[:, :, 1] * mask_gb + RI_Gbh
        RI_Guidegh = RI_Guidegrh + RI_Guidegbh
        RI_Guiderh = mosaic[:, :, 0] + RI_Rh
        RI_Guidebh = mosaic[:, :, 2] + RI_Bh
        RI_Guidegrv = mosaic[:, :, 1] * mask_gb + RI_Grv
        RI_Guidegbv = mosaic[:, :, 1] * mask_gr + RI_Gbv
        RI_Guidegv = RI_Guidegrv + RI_Guidegbv
        RI_Guiderv = mosaic[:, :, 0] + RI_Rv
        RI_Guidebv = mosaic[:, :, 2] + RI_Bv
        MLRI_Guidegrh = mosaic[:, :, 1] * mask_gr + MLRI_Grh
        MLRI_Guidegbh = mosaic[:, :, 1] * mask_gb + MLRI_Gbh
        MLRI_Guidegh = MLRI_Guidegrh + MLRI_Guidegbh
        MLRI_Guiderh = mosaic[:, :, 0] + MLRI_Rh
        MLRI_Guidebh = mosaic[:, :, 2] + MLRI_Bh
        MLRI_Guidegrv = mosaic[:, :, 1] * mask_gb + MLRI_Grv
        MLRI_Guidegbv = mosaic[:, :, 1] * mask_gr + MLRI_Gbv
        MLRI_Guidegv = MLRI_Guidegrv + MLRI_Guidegbv
        MLRI_Guiderv = mosaic[:, :, 0] + MLRI_Rv
        MLRI_Guidebv = mosaic[:, :, 2] + MLRI_Bv

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
    green = green * imask[:, :, 1] + mosaic[:, :, 1]
    return _clip(green, 0, 255)


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
    return np.array(
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


def _red_interpolation(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, h: int, v: int, eps: float) -> np.ndarray:
    F = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 4, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float64,
    )
    lap_red = _imfilter(mosaic[:, :, 0], F, "replicate")
    lap_green = _imfilter(green * mask[:, :, 0], F, "replicate")

    tentativeR = _guidedfilter_mlri(green, mosaic[:, :, 0], mask[:, :, 0], lap_green, lap_red, mask[:, :, 0], h, v, eps)
    tentativeR = _clip(tentativeR, 0, 255)
    residualR = mask[:, :, 0] * (mosaic[:, :, 0] - tentativeR)
    residualR = _imfilter(residualR, _bicubic_residual_kernel(), "replicate")
    return _clip(residualR + tentativeR, 0, 255)


def _blue_interpolation(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, h: int, v: int, eps: float) -> np.ndarray:
    F = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 4, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float64,
    )
    lap_blue = _imfilter(mosaic[:, :, 2], F, "replicate")
    lap_green = _imfilter(green * mask[:, :, 2], F, "replicate")

    tentativeB = _guidedfilter_mlri(green, mosaic[:, :, 2], mask[:, :, 2], lap_green, lap_blue, mask[:, :, 2], h, v, eps)
    tentativeB = _clip(tentativeB, 0, 255)
    residualB = mask[:, :, 2] * (mosaic[:, :, 2] - tentativeB)
    residualB = _imfilter(residualB, _bicubic_residual_kernel(), "replicate")
    return _clip(residualB + tentativeB, 0, 255)


def _inverse_mask(mask: np.ndarray) -> np.ndarray:
    imask = np.zeros_like(mask, dtype=np.float64)
    imask[:, :, 0] = (mask[:, :, 0] == 0).astype(np.float64)
    imask[:, :, 1] = (mask[:, :, 1] == 0).astype(np.float64)
    imask[:, :, 2] = (mask[:, :, 2] == 0).astype(np.float64)
    return imask


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

    RI_w2R1 = np.ones_like(mask[:, :, 0]) * 1e32
    RI_w2R2 = np.ones_like(mask[:, :, 0]) * 1e32
    MLRI_w2R1 = np.ones_like(mask[:, :, 0]) * 1e32
    MLRI_w2R2 = np.ones_like(mask[:, :, 0]) * 1e32
    RI_w2B1 = np.ones_like(mask[:, :, 0]) * 1e32
    RI_w2B2 = np.ones_like(mask[:, :, 0]) * 1e32
    MLRI_w2B1 = np.ones_like(mask[:, :, 0]) * 1e32
    MLRI_w2B2 = np.ones_like(mask[:, :, 0]) * 1e32

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

        RI_residualR1 = (mosaic[:, :, 0] - RI_tentativeR1) * mask[:, :, 0]
        RI_residualB1 = (mosaic[:, :, 2] - RI_tentativeB1) * mask[:, :, 2]
        RI_residualR2 = (mosaic[:, :, 0] - RI_tentativeR2) * mask[:, :, 0]
        RI_residualB2 = (mosaic[:, :, 2] - RI_tentativeB2) * mask[:, :, 2]
        MLRI_residualR1 = (mosaic[:, :, 0] - MLRI_tentativeR1) * mask[:, :, 0]
        MLRI_residualB1 = (mosaic[:, :, 2] - MLRI_tentativeB1) * mask[:, :, 2]
        MLRI_residualR2 = (mosaic[:, :, 0] - MLRI_tentativeR2) * mask[:, :, 0]
        MLRI_residualB2 = (mosaic[:, :, 2] - MLRI_tentativeB2) * mask[:, :, 2]

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

        RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask[:, :, 2]
        RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask[:, :, 0]
        RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask[:, :, 2]
        RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask[:, :, 0]
        MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask[:, :, 2]
        MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask[:, :, 0]
        MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask[:, :, 2]
        MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask[:, :, 0]

        RI_criR1 = (RI_Guider1 - RI_tentativeR1) * imask[:, :, 1]
        RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * imask[:, :, 1]
        RI_criR2 = (RI_Guider2 - RI_tentativeR2) * imask[:, :, 1]
        RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * imask[:, :, 1]
        MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * imask[:, :, 1]
        MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * imask[:, :, 1]
        MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * imask[:, :, 1]
        MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * imask[:, :, 1]

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
    imask = np.empty_like(mask, dtype=np.float64)
    imask[:, :, 0] = (mask[:, :, 0] == 0).astype(np.float64)
    imask[:, :, 1] = (mask[:, :, 1] == 0).astype(np.float64)
    imask[:, :, 2] = (mask[:, :, 2] == 0).astype(np.float64)
    return imask


def _red_blue_interpolation_first(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    imask = _inverse_mask(mask)

    F1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64) / 2
    Guider1 = mosaic[:, :, 0] + _imfilter(mosaic[:, :, 0], F1, "replicate") * mask[:, :, 2]
    Guideg1 = green * imask[:, :, 1]
    Guideb1 = mosaic[:, :, 2] + _imfilter(mosaic[:, :, 2], F1, "replicate") * mask[:, :, 0]
    F2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64) / 2
    Guider2 = mosaic[:, :, 0] + _imfilter(mosaic[:, :, 0], F2, "replicate") * mask[:, :, 2]
    Guideg2 = green * imask[:, :, 1]
    Guideb2 = mosaic[:, :, 2] + _imfilter(mosaic[:, :, 2], F2, "replicate") * mask[:, :, 0]

    h = 2
    v = 2
    h2 = 2
    v2 = 0
    itnum = 2

    RI_w2R1 = np.ones_like(mask[:, :, 0]) * 1e32
    RI_w2R2 = np.ones_like(mask[:, :, 0]) * 1e32
    MLRI_w2R1 = np.ones_like(mask[:, :, 0]) * 1e32
    MLRI_w2R2 = np.ones_like(mask[:, :, 0]) * 1e32
    RI_w2B1 = np.ones_like(mask[:, :, 0]) * 1e32
    RI_w2B2 = np.ones_like(mask[:, :, 0]) * 1e32
    MLRI_w2B1 = np.ones_like(mask[:, :, 0]) * 1e32
    MLRI_w2B2 = np.ones_like(mask[:, :, 0]) * 1e32

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

    for _ in range(itnum):
        RI_tentativeR1 = _guidedfilter_diagonal(RI_Guideg1, RI_Guider1, imask[:, :, 1], h, v, eps)
        RI_tentativeR2 = _guidedfilter_diagonal(RI_Guideg2, RI_Guider2, imask[:, :, 1], v, h, eps)
        RI_tentativeB1 = _guidedfilter_diagonal(RI_Guideg1, RI_Guideb1, imask[:, :, 1], h, v, eps)
        RI_tentativeB2 = _guidedfilter_diagonal(RI_Guideg2, RI_Guideb2, imask[:, :, 1], v, h, eps)

        F1 = np.array(
            [[-1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, -1]],
            dtype=np.float64,
        )
        difR = _imfilter(MLRI_Guider1, F1, "replicate")
        difG = _imfilter(MLRI_Guideg1, F1, "replicate")
        difB = _imfilter(MLRI_Guideb1, F1, "replicate")
        MLRI_tentativeR1 = _guidedfilter_mlri_diagonal(MLRI_Guideg1, MLRI_Guider1, imask[:, :, 1], difG, difR, mask[:, :, 0], h2, v2, eps)
        MLRI_tentativeB1 = _guidedfilter_mlri_diagonal(MLRI_Guideg1, MLRI_Guideb1, imask[:, :, 1], difG, difB, mask[:, :, 2], h2, v2, eps)

        F2 = np.array(
            [[0, 0, 0, 0, -1], [0, 0, 0, 0, 0], [0, 0, 2, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 0]],
            dtype=np.float64,
        )
        difR = _imfilter(MLRI_Guider2, F2, "replicate")
        difG = _imfilter(MLRI_Guideg2, F2, "replicate")
        difB = _imfilter(MLRI_Guideb2, F2, "replicate")
        MLRI_tentativeR2 = _guidedfilter_mlri_diagonal(MLRI_Guideg2, MLRI_Guider2, imask[:, :, 1], difG, difR, mask[:, :, 0], v2, h2, eps)
        MLRI_tentativeB2 = _guidedfilter_mlri_diagonal(MLRI_Guideg2, MLRI_Guideb2, imask[:, :, 1], difG, difB, mask[:, :, 2], v2, h2, eps)

        RI_residualR1 = (mosaic[:, :, 0] - RI_tentativeR1) * mask[:, :, 0]
        RI_residualB1 = (mosaic[:, :, 2] - RI_tentativeB1) * mask[:, :, 2]
        RI_residualR2 = (mosaic[:, :, 0] - RI_tentativeR2) * mask[:, :, 0]
        RI_residualB2 = (mosaic[:, :, 2] - RI_tentativeB2) * mask[:, :, 2]
        MLRI_residualR1 = (mosaic[:, :, 0] - MLRI_tentativeR1) * mask[:, :, 0]
        MLRI_residualB1 = (mosaic[:, :, 2] - MLRI_tentativeB1) * mask[:, :, 2]
        MLRI_residualR2 = (mosaic[:, :, 0] - MLRI_tentativeR2) * mask[:, :, 0]
        MLRI_residualB2 = (mosaic[:, :, 2] - MLRI_tentativeB2) * mask[:, :, 2]

        K1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64) / 2
        RI_residualR1 = _imfilter(RI_residualR1, K1, "replicate")
        RI_residualB1 = _imfilter(RI_residualB1, K1, "replicate")
        MLRI_residualR1 = _imfilter(MLRI_residualR1, K1, "replicate")
        MLRI_residualB1 = _imfilter(MLRI_residualB1, K1, "replicate")
        K2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.float64) / 2
        RI_residualR2 = _imfilter(RI_residualR2, K2, "replicate")
        RI_residualB2 = _imfilter(RI_residualB2, K2, "replicate")
        MLRI_residualR2 = _imfilter(MLRI_residualR2, K2, "replicate")
        MLRI_residualB2 = _imfilter(MLRI_residualB2, K2, "replicate")

        RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask[:, :, 2]
        RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask[:, :, 0]
        RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask[:, :, 2]
        RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask[:, :, 0]
        MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask[:, :, 2]
        MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask[:, :, 0]
        MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask[:, :, 2]
        MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask[:, :, 0]

        RI_criR1 = (RI_Guider1 - RI_tentativeR1) * imask[:, :, 1]
        RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * imask[:, :, 1]
        RI_criR2 = (RI_Guider2 - RI_tentativeR2) * imask[:, :, 1]
        RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * imask[:, :, 1]
        MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * imask[:, :, 1]
        MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * imask[:, :, 1]
        MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * imask[:, :, 1]
        MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * imask[:, :, 1]

        F1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.float64)
        RI_difcriR1 = np.abs(_imfilter(RI_criR1, F1, "replicate"))
        RI_difcriB1 = np.abs(_imfilter(RI_criB1, F1, "replicate"))
        MLRI_difcriR1 = np.abs(_imfilter(MLRI_criR1, F1, "replicate"))
        MLRI_difcriB1 = np.abs(_imfilter(MLRI_criB1, F1, "replicate"))
        F2 = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]], dtype=np.float64)
        RI_difcriR2 = np.abs(_imfilter(RI_criR2, F2, "replicate"))
        RI_difcriB2 = np.abs(_imfilter(RI_criB2, F2, "replicate"))
        MLRI_difcriR2 = np.abs(_imfilter(MLRI_criR2, F2, "replicate"))
        MLRI_difcriB2 = np.abs(_imfilter(MLRI_criB2, F2, "replicate"))

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

        Fg = _gaussian_kernel((5, 5), 2)
        M1 = _imfilter(imask[:, :, 1], Fg, "replicate")
        RI_criR1 = _imfilter(RI_criR1, Fg, "replicate") / M1 * imask[:, :, 1]
        MLRI_criR1 = _imfilter(MLRI_criR1, Fg, "replicate") / M1 * imask[:, :, 1]
        RI_criB1 = _imfilter(RI_criB1, Fg, "replicate") / M1 * imask[:, :, 1]
        MLRI_criB1 = _imfilter(MLRI_criB1, Fg, "replicate") / M1 * imask[:, :, 1]
        RI_difcriR1 = _imfilter(RI_difcriR1, Fg, "replicate") / M1 * imask[:, :, 1]
        MLRI_difcriR1 = _imfilter(MLRI_difcriR1, Fg, "replicate") / M1 * imask[:, :, 1]
        RI_difcriB1 = _imfilter(RI_difcriB1, Fg, "replicate") / M1 * imask[:, :, 1]
        MLRI_difcriB1 = _imfilter(MLRI_difcriB1, Fg, "replicate") / M1 * imask[:, :, 1]
        M2 = _imfilter(imask[:, :, 1], Fg, "replicate")
        RI_criR2 = _imfilter(RI_criR2, Fg, "replicate") / M2 * imask[:, :, 1]
        MLRI_criR2 = _imfilter(MLRI_criR2, Fg, "replicate") / M2 * imask[:, :, 1]
        RI_criB2 = _imfilter(RI_criB2, Fg, "replicate") / M2 * imask[:, :, 1]
        MLRI_criB2 = _imfilter(MLRI_criB2, Fg, "replicate") / M2 * imask[:, :, 1]
        RI_difcriR2 = _imfilter(RI_difcriR2, Fg, "replicate") / M2 * imask[:, :, 1]
        MLRI_difcriR2 = _imfilter(MLRI_difcriR2, Fg, "replicate") / M2 * imask[:, :, 1]
        RI_difcriB2 = _imfilter(RI_difcriB2, Fg, "replicate") / M2 * imask[:, :, 1]
        MLRI_difcriB2 = _imfilter(MLRI_difcriB2, Fg, "replicate") / M2 * imask[:, :, 1]

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
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    imask = _inverse_mask(mask)

    F1 = np.array([[0.5, 0, 0.5]], dtype=np.float64)
    Guider1 = red + _imfilter(red, F1, "replicate") * mask[:, :, 1]
    Guideg1 = green
    Guideb1 = blue + _imfilter(blue, F1, "replicate") * mask[:, :, 1]
    F2 = F1.T
    Guider2 = red + _imfilter(red, F2, "replicate") * mask[:, :, 1]
    Guideg2 = green
    Guideb2 = blue + _imfilter(blue, F2, "replicate") * mask[:, :, 1]

    h = 2
    v = 2
    h2 = 2
    v2 = 0
    itnum = 2
    base = np.ones_like(mask[:, :, 0])

    RI_w2R1 = base * 1e32
    RI_w2R2 = base * 1e32
    MLRI_w2R1 = base * 1e32
    MLRI_w2R2 = base * 1e32
    RI_w2B1 = base * 1e32
    RI_w2B2 = base * 1e32
    MLRI_w2B1 = base * 1e32
    MLRI_w2B2 = base * 1e32

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

    for _ in range(itnum):
        M = np.ones_like(mask[:, :, 0])
        RI_tentativeR1 = _guidedfilter(RI_Guideg1, RI_Guider1, M, h, v, eps)
        RI_tentativeB1 = _guidedfilter(RI_Guideg1, RI_Guideb1, M, h, v, eps)
        RI_tentativeR2 = _guidedfilter(RI_Guideg2, RI_Guider2, M, v, h, eps)
        RI_tentativeB2 = _guidedfilter(RI_Guideg2, RI_Guideb2, M, v, h, eps)

        F1 = np.array([[-1, 0, 2, 0, -1]], dtype=np.float64)
        difR = _imfilter(MLRI_Guider1, F1, "replicate")
        difG = _imfilter(MLRI_Guideg1, F1, "replicate")
        difB = _imfilter(MLRI_Guideb1, F1, "replicate")
        MLRI_tentativeR1 = _guidedfilter_mlri(MLRI_Guideg1, MLRI_Guider1, M, difG, difR, imask[:, :, 1], h2, v2, eps)
        MLRI_tentativeB1 = _guidedfilter_mlri(MLRI_Guideg1, MLRI_Guideb1, M, difG, difB, imask[:, :, 1], h2, v2, eps)
        F2 = F1.T
        difR = _imfilter(MLRI_Guider2, F2, "replicate")
        difG = _imfilter(MLRI_Guideg2, F2, "replicate")
        difB = _imfilter(MLRI_Guideb2, F2, "replicate")
        MLRI_tentativeR2 = _guidedfilter_mlri(MLRI_Guideg2, MLRI_Guider2, M, difG, difR, imask[:, :, 1], v2, h2, eps)
        MLRI_tentativeB2 = _guidedfilter_mlri(MLRI_Guideg2, MLRI_Guideb2, M, difG, difB, imask[:, :, 1], v2, h2, eps)

        RI_residualR1 = (red - RI_tentativeR1) * imask[:, :, 1]
        RI_residualB1 = (blue - RI_tentativeB1) * imask[:, :, 1]
        RI_residualR2 = (red - RI_tentativeR2) * imask[:, :, 1]
        RI_residualB2 = (blue - RI_tentativeB2) * imask[:, :, 1]
        MLRI_residualR1 = (red - MLRI_tentativeR1) * imask[:, :, 1]
        MLRI_residualB1 = (blue - MLRI_tentativeB1) * imask[:, :, 1]
        MLRI_residualR2 = (red - MLRI_tentativeR2) * imask[:, :, 1]
        MLRI_residualB2 = (blue - MLRI_tentativeB2) * imask[:, :, 1]

        K1 = np.array([[0.5, 0, 0.5]], dtype=np.float64)
        RI_residualR1 = _imfilter(RI_residualR1, K1, "replicate")
        RI_residualB1 = _imfilter(RI_residualB1, K1, "replicate")
        MLRI_residualR1 = _imfilter(MLRI_residualR1, K1, "replicate")
        MLRI_residualB1 = _imfilter(MLRI_residualB1, K1, "replicate")
        K2 = K1.T
        RI_residualR2 = _imfilter(RI_residualR2, K2, "replicate")
        RI_residualB2 = _imfilter(RI_residualB2, K2, "replicate")
        MLRI_residualR2 = _imfilter(MLRI_residualR2, K2, "replicate")
        MLRI_residualB2 = _imfilter(MLRI_residualB2, K2, "replicate")

        RI_R1 = (RI_tentativeR1 + RI_residualR1) * mask[:, :, 1]
        RI_B1 = (RI_tentativeB1 + RI_residualB1) * mask[:, :, 1]
        RI_R2 = (RI_tentativeR2 + RI_residualR2) * mask[:, :, 1]
        RI_B2 = (RI_tentativeB2 + RI_residualB2) * mask[:, :, 1]
        MLRI_R1 = (MLRI_tentativeR1 + MLRI_residualR1) * mask[:, :, 1]
        MLRI_B1 = (MLRI_tentativeB1 + MLRI_residualB1) * mask[:, :, 1]
        MLRI_R2 = (MLRI_tentativeR2 + MLRI_residualR2) * mask[:, :, 1]
        MLRI_B2 = (MLRI_tentativeB2 + MLRI_residualB2) * mask[:, :, 1]

        RI_criR1 = (RI_Guider1 - RI_tentativeR1) * M
        RI_criB1 = (RI_Guideb1 - RI_tentativeB1) * M
        RI_criR2 = (RI_Guider2 - RI_tentativeR2) * M
        RI_criB2 = (RI_Guideb2 - RI_tentativeB2) * M
        MLRI_criR1 = (MLRI_Guider1 - MLRI_tentativeR1) * M
        MLRI_criB1 = (MLRI_Guideb1 - MLRI_tentativeB1) * M
        MLRI_criR2 = (MLRI_Guider2 - MLRI_tentativeR2) * M
        MLRI_criB2 = (MLRI_Guideb2 - MLRI_tentativeB2) * M

        F1 = np.array([[-1, 0, 1]], dtype=np.float64)
        RI_difcriR1 = np.abs(_imfilter(RI_criR1, F1, "replicate"))
        RI_difcriB1 = np.abs(_imfilter(RI_criB1, F1, "replicate"))
        MLRI_difcriR1 = np.abs(_imfilter(MLRI_criR1, F1, "replicate"))
        MLRI_difcriB1 = np.abs(_imfilter(MLRI_criB1, F1, "replicate"))
        F2 = F1.T
        RI_difcriR2 = np.abs(_imfilter(RI_criR2, F2, "replicate"))
        RI_difcriB2 = np.abs(_imfilter(RI_criB2, F2, "replicate"))
        MLRI_difcriR2 = np.abs(_imfilter(MLRI_criR2, F2, "replicate"))
        MLRI_difcriB2 = np.abs(_imfilter(MLRI_criB2, F2, "replicate"))

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

        Fg = _gaussian_kernel((5, 5), 2)
        RI_criR1 = _imfilter(RI_criR1, Fg, "replicate")
        MLRI_criR1 = _imfilter(MLRI_criR1, Fg, "replicate")
        RI_criB1 = _imfilter(RI_criB1, Fg, "replicate")
        MLRI_criB1 = _imfilter(MLRI_criB1, Fg, "replicate")
        RI_difcriR1 = _imfilter(RI_difcriR1, Fg, "replicate")
        MLRI_difcriR1 = _imfilter(MLRI_difcriR1, Fg, "replicate")
        RI_difcriB1 = _imfilter(RI_difcriB1, Fg, "replicate")
        MLRI_difcriB1 = _imfilter(MLRI_difcriB1, Fg, "replicate")
        RI_criR2 = _imfilter(RI_criR2, Fg, "replicate")
        MLRI_criR2 = _imfilter(MLRI_criR2, Fg, "replicate")
        RI_criB2 = _imfilter(RI_criB2, Fg, "replicate")
        MLRI_criB2 = _imfilter(MLRI_criB2, Fg, "replicate")
        RI_difcriR2 = _imfilter(RI_difcriR2, Fg, "replicate")
        MLRI_difcriR2 = _imfilter(MLRI_difcriR2, Fg, "replicate")
        RI_difcriB2 = _imfilter(RI_difcriB2, Fg, "replicate")
        MLRI_difcriB2 = _imfilter(MLRI_difcriB2, Fg, "replicate")

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
