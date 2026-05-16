from __future__ import annotations

import math

import numpy as np

from .matlab_compat import filter2_valid, gaussian_kernel


def _crop_psnr(x: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return x
    # MATLAB X(b:size-b,...) is 1-based inclusive. For b=10 this starts at
    # zero-based index 9 and excludes the final 10 pixels.
    return x[border - 1 : x.shape[0] - border, border - 1 : x.shape[1] - border, ...]


def psnr(
    reference: np.ndarray,
    estimate: np.ndarray,
    peak: float = 255.0,
    border: int = 0,
    b: int | None = None,
) -> np.ndarray:
    if b is not None:
        border = b
    x = _crop_psnr(np.asarray(reference, dtype=np.float64), border)
    y = _crop_psnr(np.asarray(estimate, dtype=np.float64), border)
    dif = (x - y) ** 2
    mse = np.mean(dif, axis=(0, 1)) + 1e-32
    return 10.0 * np.log10((peak * peak) / mse)


def cpsnr(
    reference: np.ndarray,
    estimate: np.ndarray,
    peak: float = 255.0,
    border: int = 0,
    b: int | None = None,
) -> float:
    if b is not None:
        border = b
    x = _crop_psnr(np.asarray(reference, dtype=np.float64), border)
    y = _crop_psnr(np.asarray(estimate, dtype=np.float64), border)
    mse = np.mean((x - y) ** 2) + 1e-32
    return float(10.0 * np.log10((peak * peak) / mse))


def ssim_index(
    img1: np.ndarray,
    img2: np.ndarray,
    k: tuple[float, float] = (0.01, 0.03),
    window: np.ndarray | None = None,
    peak: float = 255.0,
) -> tuple[float, np.ndarray]:
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    if img1.shape != img2.shape or img1.ndim != 2:
        return -math.inf, -math.inf
    if window is None:
        if img1.shape[0] < 11 or img1.shape[1] < 11:
            return -math.inf, -math.inf
        window = gaussian_kernel((11, 11), 1.5)
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        return -math.inf, -math.inf
    if window.size < 4 or window.shape[0] > img1.shape[0] or window.shape[1] > img1.shape[1]:
        return -math.inf, -math.inf
    if len(k) != 2 or k[0] < 0 or k[1] < 0:
        return -math.inf, -math.inf
    window = window / np.sum(window)

    c1 = (k[0] * peak) ** 2
    c2 = (k[1] * peak) ** 2

    mu1 = filter2_valid(window, img1)
    mu2 = filter2_valid(window, img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2_valid(window, img1 * img1) - mu1_sq
    sigma2_sq = filter2_valid(window, img2 * img2) - mu2_sq
    sigma12 = filter2_valid(window, img1 * img2) - mu1_mu2

    if c1 > 0 and c2 > 0:
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
    else:
        numerator1 = 2 * mu1_mu2 + c1
        numerator2 = 2 * sigma12 + c2
        denominator1 = mu1_sq + mu2_sq + c1
        denominator2 = sigma1_sq + sigma2_sq + c2
        ssim_map = np.ones_like(mu1)
        index = denominator1 * denominator2 > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (
            denominator1[index] * denominator2[index]
        )
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]
    return float(np.mean(ssim_map)), ssim_map


def _crop_ssim_matlab_imcrop(x: np.ndarray) -> np.ndarray:
    # imcrop(I,[11,11,w-10,h-10]) in MATLAB uses 1-based spatial coordinates
    # and inclusive extents. For these images it maps to zero-based 10:height,
    # 10:width, matching the script's intended 10-pixel top/left crop.
    return x[10:, 10:, ...]


def ssim(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    x = _crop_ssim_matlab_imcrop(np.asarray(reference, dtype=np.float64))
    y = _crop_ssim_matlab_imcrop(np.asarray(estimate, dtype=np.float64))
    vals = np.array([ssim_index(x[:, :, c], y[:, :, c])[0] for c in range(3)], dtype=np.float64)
    return np.append(vals, np.mean(vals))
