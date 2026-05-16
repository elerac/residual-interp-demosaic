from __future__ import annotations

import numpy as np

from demosaic_reference.bayer import mask_gr_gb
from demosaic_reference.matlab_compat import boxfilter, clip, gaussian_kernel, imfilter


def _guidedfilter_ri(i: np.ndarray, p: np.ndarray, m: np.ndarray, h: int, v: int, eps: float) -> np.ndarray:
    th = 0.00001 * 255 * 255
    hei, wid = i.shape
    n = boxfilter(m, h, v)
    n[n == 0] = 1
    n2 = boxfilter(np.ones((hei, wid)), h, v)

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
) -> np.ndarray:
    th = 0.00001 * 255 * 255
    hei, wid = i.shape
    n = boxfilter(m, h, v)
    n[n == 0] = 1
    n2 = boxfilter(np.ones((hei, wid)), h, v)

    mean_ip = boxfilter(i * p * m, h, v) / n
    mean_ii = boxfilter(i * i * m, h, v) / n
    mean_ii[mean_ii < th] = th
    a = mean_ip / (mean_ii + eps)

    n3 = boxfilter(mask, h, v)
    n3[n3 == 0] = 1
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
) -> np.ndarray:
    n = boxfilter(m, h, v)
    n[n == 0] = 1

    mean_ip = boxfilter(i * p * m, h, v) / n
    mean_ii = boxfilter(i * i * m, h, v) / n
    a = mean_ip / (mean_ii + eps)

    n3 = boxfilter(mask, h, v)
    n3[n3 == 0] = 1
    mean_g = boxfilter(g * mask, h, v) / n3
    mean_r = boxfilter(r * mask, h, v) / n3
    b = mean_r - a * mean_g

    dif = (
        boxfilter(g * g * mask, h, v) * a * a
        + b * b * n3
        + boxfilter(r * r * mask, h, v)
        + 2 * a * b * boxfilter(g * mask, h, v)
        - 2 * b * boxfilter(r * mask, h, v)
        - 2 * a * boxfilter(r * g * mask, h, v)
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
    imask = (mask == 0).astype(np.float64)
    rawq = np.sum(mosaic, axis=2)
    mask_gr, mask_gb = mask_gr_gb(rawq.shape, pattern)

    kh = np.array([[0.5, 0.0, 0.5]])
    kv = kh.T
    rawh = imfilter(rawq, kh)
    rawv = imfilter(rawq, kv)

    guide_gh = mosaic[:, :, 1] + rawh * mask[:, :, 0] + rawh * mask[:, :, 2]
    guide_rh = mosaic[:, :, 0] + rawh * mask_gr
    guide_bh = mosaic[:, :, 2] + rawh * mask_gb
    guide_gv = mosaic[:, :, 1] + rawv * mask[:, :, 0] + rawv * mask[:, :, 2]
    guide_rv = mosaic[:, :, 0] + rawv * mask_gb
    guide_bv = mosaic[:, :, 2] + rawv * mask_gr

    h, v, eps = 5, 0, 0.0
    tentative_rh = _guidedfilter_ri(guide_gh, mosaic[:, :, 0], mask[:, :, 0], h, v, eps)
    tentative_grh = _guidedfilter_ri(guide_rh, mosaic[:, :, 1] * mask_gr, mask_gr, h, v, eps)
    tentative_gbh = _guidedfilter_ri(guide_bh, mosaic[:, :, 1] * mask_gb, mask_gb, h, v, eps)
    tentative_bh = _guidedfilter_ri(guide_gh, mosaic[:, :, 2], mask[:, :, 2], h, v, eps)
    tentative_rv = _guidedfilter_ri(guide_gv, mosaic[:, :, 0], mask[:, :, 0], v, h, eps)
    tentative_grv = _guidedfilter_ri(guide_rv, mosaic[:, :, 1] * mask_gb, mask_gb, v, h, eps)
    tentative_gbv = _guidedfilter_ri(guide_bv, mosaic[:, :, 1] * mask_gr, mask_gr, v, h, eps)
    tentative_bv = _guidedfilter_ri(guide_gv, mosaic[:, :, 2], mask[:, :, 2], v, h, eps)

    residual_grh = (mosaic[:, :, 1] - tentative_grh) * mask_gr
    residual_gbh = (mosaic[:, :, 1] - tentative_gbh) * mask_gb
    residual_rh = (mosaic[:, :, 0] - tentative_rh) * mask[:, :, 0]
    residual_bh = (mosaic[:, :, 2] - tentative_bh) * mask[:, :, 2]
    residual_grv = (mosaic[:, :, 1] - tentative_grv) * mask_gb
    residual_gbv = (mosaic[:, :, 1] - tentative_gbv) * mask_gr
    residual_rv = (mosaic[:, :, 0] - tentative_rv) * mask[:, :, 0]
    residual_bv = (mosaic[:, :, 2] - tentative_bv) * mask[:, :, 2]

    residual_grh = imfilter(residual_grh, kh)
    residual_gbh = imfilter(residual_gbh, kh)
    residual_rh = imfilter(residual_rh, kh)
    residual_bh = imfilter(residual_bh, kh)
    residual_grv = imfilter(residual_grv, kv)
    residual_gbv = imfilter(residual_gbv, kv)
    residual_rv = imfilter(residual_rv, kv)
    residual_bv = imfilter(residual_bv, kv)

    grh = (tentative_grh + residual_grh) * mask[:, :, 0]
    gbh = (tentative_gbh + residual_gbh) * mask[:, :, 2]
    rh = (tentative_rh + residual_rh) * mask_gr
    bh = (tentative_bh + residual_bh) * mask_gb
    grv = (tentative_grv + residual_grv) * mask[:, :, 0]
    gbv = (tentative_gbv + residual_gbv) * mask[:, :, 2]
    rv = (tentative_rv + residual_rv) * mask_gb
    bv = (tentative_bv + residual_bv) * mask_gr

    difh = mosaic[:, :, 1] + grh + gbh - mosaic[:, :, 0] - mosaic[:, :, 2] - rh - bh
    difv = mosaic[:, :, 1] + grv + gbv - mosaic[:, :, 0] - mosaic[:, :, 2] - rv - bv

    kh = np.array([[1.0, 0.0, -1.0]])
    kv = kh.T
    difh2 = np.abs(imfilter(difh, kh))
    difv2 = np.abs(imfilter(difv, kv))
    k = np.ones((5, 5))
    wh = imfilter(difh2, k)
    wv = imfilter(difv2, k)
    kw = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    ke = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
    kn = kw.T
    ks = ke.T
    ww = 1.0 / (imfilter(wh, kw) ** 2 + 1e-32)
    we = 1.0 / (imfilter(wh, ke) ** 2 + 1e-32)
    wn = 1.0 / (imfilter(wv, kn) ** 2 + 1e-32)
    ws = 1.0 / (imfilter(wv, ks) ** 2 + 1e-32)

    hwin = gaussian_kernel((1, 9), sigma)
    ke = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=np.float64) * hwin
    kw = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.float64) * hwin
    ke = ke / np.sum(ke, axis=1, keepdims=True)
    kw = kw / np.sum(kw, axis=1, keepdims=True)
    ks = ke.T
    kn = kw.T
    difn = imfilter(difv, kn)
    difs = imfilter(difv, ks)
    difw = imfilter(difh, kw)
    dife = imfilter(difh, ke)
    wt = ww + we + wn + ws
    dif = (wn * difn + ws * difs + ww * difw + we * dife) / wt
    green = dif + rawq
    green = green * imask[:, :, 1] + rawq * mask[:, :, 1]
    return clip(green, 0, 255)


def _mlri_green_interpolation(
    mosaic: np.ndarray,
    mask: np.ndarray,
    pattern: str,
    sigma: float,
    eps: float,
    weighted: bool,
) -> np.ndarray:
    imask = (mask == 0).astype(np.float64)
    rawq = np.sum(mosaic, axis=2)
    mask_gr, mask_gb = mask_gr_gb(rawq.shape, pattern)
    gf = _guidedfilter_mlri_weighted if weighted else _guidedfilter_mlri

    kh = np.array([[0.5, 0.0, 0.5]])
    kv = kh.T
    rawh = imfilter(rawq, kh)
    rawv = imfilter(rawq, kv)
    guide_gh = mosaic[:, :, 1] + rawh * mask[:, :, 0] + rawh * mask[:, :, 2]
    guide_rh = mosaic[:, :, 0] + rawh * mask_gr
    guide_bh = mosaic[:, :, 2] + rawh * mask_gb
    guide_gv = mosaic[:, :, 1] + rawv * mask[:, :, 0] + rawv * mask[:, :, 2]
    guide_rv = mosaic[:, :, 0] + rawv * mask_gb
    guide_bv = mosaic[:, :, 2] + rawv * mask_gr

    h, v = 3, 3
    f = np.array([[-1.0, 0.0, 2.0, 0.0, -1.0]])
    dif_r = imfilter(mosaic[:, :, 0], f)
    dif_gr = imfilter(guide_gh * mask[:, :, 0], f)
    tentative_rh = gf(guide_gh, mosaic[:, :, 0], mask[:, :, 0], dif_gr, dif_r, mask[:, :, 0], h, v, eps)

    dif_gr = imfilter(mosaic[:, :, 1] * mask_gr, f)
    dif_r = imfilter(guide_rh * mask_gr, f)
    tentative_grh = gf(guide_rh, mosaic[:, :, 1] * mask_gr, mask_gr, dif_r, dif_gr, mask_gr, h, v, eps)

    dif_b = imfilter(mosaic[:, :, 2], f)
    dif_gb = imfilter(guide_gh * mask[:, :, 2], f)
    tentative_bh = gf(guide_gh, mosaic[:, :, 2], mask[:, :, 2], dif_gb, dif_b, mask[:, :, 2], h, v, eps)

    dif_gb = imfilter(mosaic[:, :, 1] * mask_gb, f)
    dif_b = imfilter(guide_bh * mask_gb, f)
    tentative_gbh = gf(guide_bh, mosaic[:, :, 1] * mask_gb, mask_gb, dif_b, dif_gb, mask_gb, h, v, eps)

    f = f.T
    dif_r = imfilter(mosaic[:, :, 0], f)
    dif_gr = imfilter(guide_gv * mask[:, :, 0], f)
    tentative_rv = gf(guide_gv, mosaic[:, :, 0], mask[:, :, 0], dif_gr, dif_r, mask[:, :, 0], v, h, eps)

    dif_gr = imfilter(mosaic[:, :, 1] * mask_gb, f)
    dif_r = imfilter(guide_rv * mask_gb, f)
    tentative_grv = gf(guide_rv, mosaic[:, :, 1] * mask_gb, mask_gb, dif_r, dif_gr, mask_gb, v, h, eps)

    dif_b = imfilter(mosaic[:, :, 2], f)
    dif_gb = imfilter(guide_gv * mask[:, :, 2], f)
    tentative_bv = gf(guide_gv, mosaic[:, :, 2], mask[:, :, 2], dif_gb, dif_b, mask[:, :, 2], v, h, eps)

    dif_gb = imfilter(mosaic[:, :, 1] * mask_gr, f)
    dif_b = imfilter(guide_bv * mask_gr, f)
    tentative_gbv = gf(guide_bv, mosaic[:, :, 1] * mask_gr, mask_gr, dif_b, dif_gb, mask_gr, v, h, eps)

    tentative_grh = clip(tentative_grh, 0, 255)
    tentative_grv = clip(tentative_grv, 0, 255)
    tentative_gbh = clip(tentative_gbh, 0, 255)
    tentative_gbv = clip(tentative_gbv, 0, 255)
    tentative_rh = clip(tentative_rh, 0, 255)
    tentative_rv = clip(tentative_rv, 0, 255)
    tentative_bh = clip(tentative_bh, 0, 255)
    tentative_bv = clip(tentative_bv, 0, 255)

    residual_grh = (mosaic[:, :, 1] - tentative_grh) * mask_gr
    residual_gbh = (mosaic[:, :, 1] - tentative_gbh) * mask_gb
    residual_rh = (mosaic[:, :, 0] - tentative_rh) * mask[:, :, 0]
    residual_bh = (mosaic[:, :, 2] - tentative_bh) * mask[:, :, 2]
    residual_grv = (mosaic[:, :, 1] - tentative_grv) * mask_gb
    residual_gbv = (mosaic[:, :, 1] - tentative_gbv) * mask_gr
    residual_rv = (mosaic[:, :, 0] - tentative_rv) * mask[:, :, 0]
    residual_bv = (mosaic[:, :, 2] - tentative_bv) * mask[:, :, 2]

    kh = np.array([[0.5, 0.0, 0.5]])
    kv = kh.T
    residual_grh = imfilter(residual_grh, kh)
    residual_gbh = imfilter(residual_gbh, kh)
    residual_rh = imfilter(residual_rh, kh)
    residual_bh = imfilter(residual_bh, kh)
    residual_grv = imfilter(residual_grv, kv)
    residual_gbv = imfilter(residual_gbv, kv)
    residual_rv = imfilter(residual_rv, kv)
    residual_bv = imfilter(residual_bv, kv)

    grh = clip((tentative_grh + residual_grh) * mask[:, :, 0], 0, 255)
    gbh = clip((tentative_gbh + residual_gbh) * mask[:, :, 2], 0, 255)
    rh = clip((tentative_rh + residual_rh) * mask_gr, 0, 255)
    bh = clip((tentative_bh + residual_bh) * mask_gb, 0, 255)
    grv = clip((tentative_grv + residual_grv) * mask[:, :, 0], 0, 255)
    gbv = clip((tentative_gbv + residual_gbv) * mask[:, :, 2], 0, 255)
    rv = clip((tentative_rv + residual_rv) * mask_gb, 0, 255)
    bv = clip((tentative_bv + residual_bv) * mask_gr, 0, 255)

    difh = mosaic[:, :, 1] + grh + gbh - mosaic[:, :, 0] - mosaic[:, :, 2] - rh - bh
    difv = mosaic[:, :, 1] + grv + gbv - mosaic[:, :, 0] - mosaic[:, :, 2] - rv - bv

    kh = np.array([[1.0, 0.0, -1.0]])
    kv = kh.T
    difh2 = np.abs(imfilter(difh, kh))
    difv2 = np.abs(imfilter(difv, kv))
    k = np.ones((3, 3))
    wh = imfilter(difh2, k)
    wv = imfilter(difv2, k)
    kw = np.array([[1.0, 0.0, 0.0]])
    ke = np.array([[0.0, 0.0, 1.0]])
    kn = kw.T
    ks = ke.T
    ww = 1.0 / (imfilter(wh, kw) ** 2 + 1e-2)
    we = 1.0 / (imfilter(wh, ke) ** 2 + 1e-2)
    wn = 1.0 / (imfilter(wv, kn) ** 2 + 1e-2)
    ws = 1.0 / (imfilter(wv, ks) ** 2 + 1e-2)

    hwin = gaussian_kernel((1, 9), sigma)
    ke = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=np.float64) * hwin
    kw = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.float64) * hwin
    ke = ke / np.sum(ke, axis=1, keepdims=True)
    kw = kw / np.sum(kw, axis=1, keepdims=True)
    ks = ke.T
    kn = kw.T
    difn = imfilter(difv, kn)
    difs = imfilter(difv, ks)
    difw = imfilter(difh, kw)
    dife = imfilter(difh, ke)
    wt = ww + we + wn + ws
    dif = (wn * difn + ws * difs + ww * difw + we * dife) / wt
    green = dif + rawq
    return green * imask[:, :, 1] + rawq * mask[:, :, 1]


def _ri_red_blue(green: np.ndarray, mosaic: np.ndarray, mask: np.ndarray, channel: int) -> np.ndarray:
    h, v, eps = 5, 5, 0.0
    tentative = _guidedfilter_ri(green, mosaic[:, :, channel], mask[:, :, channel], h, v, eps)
    residual = (mosaic[:, :, channel] - tentative) * mask[:, :, channel]
    kernel = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    return imfilter(residual, kernel) + tentative


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
    lap = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 4, 0, -1],
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float64,
    )
    lap_color = imfilter(mosaic[:, :, channel], lap)
    lap_green = imfilter(green * mask[:, :, channel], lap)
    tentative = gf(green, mosaic[:, :, channel], mask[:, :, channel], lap_green, lap_color, mask[:, :, channel], h, v, eps)
    tentative = clip(tentative, 0, 255)
    residual = mask[:, :, channel] * (mosaic[:, :, channel] - tentative)
    kernel = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    return imfilter(residual, kernel) + tentative


def demosaic_ri(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    green = _ri_green_interpolation(mosaic, mask, pattern, sigma=1.0)
    red = _ri_red_blue(green, mosaic, mask, 0)
    blue = _ri_red_blue(green, mosaic, mask, 2)
    return np.dstack([red, green, blue])


def demosaic_mlri(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    green = _mlri_green_interpolation(mosaic, mask, pattern, sigma=1.4, eps=0.0, weighted=False)
    green = clip(green, 0, 255)
    red = clip(_mlri_red_blue(green, mosaic, mask, 0, eps=0.0, weighted=False), 0, 255)
    blue = clip(_mlri_red_blue(green, mosaic, mask, 2, eps=0.0, weighted=False), 0, 255)
    return np.dstack([red, green, blue])


def demosaic_mlri2(mosaic: np.ndarray, mask: np.ndarray, pattern: str) -> np.ndarray:
    eps = 1e-32
    green = _mlri_green_interpolation(mosaic, mask, pattern, sigma=1.0, eps=eps, weighted=True)
    green = clip(green, 0, 255)
    red = clip(_mlri_red_blue(green, mosaic, mask, 0, eps=eps, weighted=True), 0, 255)
    blue = clip(_mlri_red_blue(green, mosaic, mask, 2, eps=eps, weighted=True), 0, 255)
    return np.dstack([red, green, blue])
