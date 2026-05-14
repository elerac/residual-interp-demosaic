import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demosaic import boxfilter, clip, cpsnr, imfilter, mosaic_bayer, parse_code, psnr, ssim_index


@pytest.mark.parametrize(
    ("code", "pattern", "algorithm"),
    [
        ("COLOR_BayerRGGB2BGR_RI", "rggb", "RI"),
        ("COLOR_BayerGRBG2BGR_MLRI", "grbg", "MLRI"),
        ("COLOR_BayerGBRG2BGR_MLRI2", "gbrg", "MLRI2"),
        ("COLOR_BayerBGGR2BGR_ARI", "bggr", "ARI"),
        ("COLOR_BayerBGGR2BGR_ARI2", "bggr", "ARI2"),
    ],
)
def test_parse_code(code, pattern, algorithm):
    parsed = parse_code(code)

    assert parsed.pattern == pattern
    assert parsed.algorithm == algorithm
    assert parsed == (pattern, algorithm)


def test_parse_code_rejects_unsupported_names():
    with pytest.raises(ValueError):
        parse_code("COLOR_BayerRGGB2RGB_RI")


def test_mosaic_bayer_matches_matlab_sparse_rgb_mask():
    rgb = np.arange(4 * 4 * 3).reshape(4, 4, 3)

    mosaic, mask = mosaic_bayer(rgb, "grbg")

    expected_mask = np.zeros_like(rgb)
    expected_mask[0::2, 0::2, 1] = 1
    expected_mask[0::2, 1::2, 0] = 1
    expected_mask[1::2, 0::2, 2] = 1
    expected_mask[1::2, 1::2, 1] = 1
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(mosaic, rgb * expected_mask)


def test_imfilter_uses_correlation_and_replicate_padding():
    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    kernel = np.array([[1, 2, 3], [0, 0, 0], [-1, -2, -3]], dtype=float)

    filtered = imfilter(image, kernel)

    expected = np.array([[-18, -18, -18], [-18, -18, -18]], dtype=float)
    np.testing.assert_allclose(filtered, expected)


def test_boxfilter_matches_matlab_edge_truncated_sums():
    image = np.arange(1, 17).reshape(4, 4)

    filtered = boxfilter(image, h=1, v=1)

    expected = np.array(
        [
            [14, 24, 30, 22],
            [33, 54, 63, 45],
            [57, 90, 99, 69],
            [46, 72, 78, 54],
        ]
    )
    np.testing.assert_array_equal(filtered, expected)


def test_boxfilter_preserves_matlab_zero_radius_behavior():
    image = np.arange(1, 10).reshape(3, 3)

    np.testing.assert_array_equal(boxfilter(image, h=0, v=0), np.zeros_like(image))


def test_clip_matches_matlab_bounds():
    image = np.array([-1, 0, 5, 10, 11])

    np.testing.assert_array_equal(clip(image, 0, 10), np.array([0, 0, 5, 10, 10]))


def test_psnr_and_cpsnr_use_matlab_border_crop_quirk():
    x = np.zeros((5, 5, 3), dtype=float)
    y = np.full_like(x, 100)
    y[1:3, 1:3, :] = 1

    per_channel = psnr(x, y, peak=1, b=2)
    combined = cpsnr(x, y, peak=1, b=2)

    np.testing.assert_allclose(per_channel, np.zeros(3), atol=1e-12)
    assert combined == pytest.approx(0)


def test_identical_images_have_finite_matlab_psnr_due_to_epsilon():
    image = np.zeros((2, 2, 3), dtype=float)

    per_channel = psnr(image, image)
    combined = cpsnr(image, image)

    expected = 10 * math.log10(255 * 255 / 1e-32)
    np.testing.assert_allclose(per_channel, np.full(3, expected))
    assert combined == pytest.approx(expected)


def test_ssim_index_defaults_for_identical_images():
    image = np.arange(12 * 12, dtype=float).reshape(12, 12)

    value, ssim_map = ssim_index(image, image)

    assert value == pytest.approx(1)
    assert ssim_map.shape == (2, 2)
    np.testing.assert_allclose(ssim_map, np.ones((2, 2)))


def test_ssim_index_returns_negative_infinity_for_too_small_default_window():
    value, ssim_map = ssim_index(np.zeros((10, 10)), np.zeros((10, 10)))

    assert value == -math.inf
    assert ssim_map == -math.inf
