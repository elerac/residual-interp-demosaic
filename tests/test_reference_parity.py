import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demosaic
import demosaic_reference
from demosaic import bayer_mask, mask_gr_gb, mosaic_bayer, parse_code
from demosaic.algorithms import (
    demosaic_ari,
    demosaic_ari2,
    demosaic_mlri,
    demosaic_mlri2,
    demosaic_ri,
)
from demosaic_reference import (
    bayer_mask as reference_bayer_mask,
)
from demosaic_reference import (
    mask_gr_gb as reference_mask_gr_gb,
)
from demosaic_reference import (
    mosaic_bayer as reference_mosaic_bayer,
)
from demosaic_reference import (
    parse_code as reference_parse_code,
)
from demosaic_reference.algorithms import (
    demosaic_ari as reference_demosaic_ari,
)
from demosaic_reference.algorithms import (
    demosaic_ari2 as reference_demosaic_ari2,
)
from demosaic_reference.algorithms import (
    demosaic_mlri as reference_demosaic_mlri,
)
from demosaic_reference.algorithms import (
    demosaic_mlri2 as reference_demosaic_mlri2,
)
from demosaic_reference.algorithms import (
    demosaic_ri as reference_demosaic_ri,
)


PATTERNS = ("rggb", "grbg", "gbrg", "bggr")
ALGORITHMS = {
    "RI": (demosaic_ri, reference_demosaic_ri),
    "MLRI": (demosaic_mlri, reference_demosaic_mlri),
    "MLRI2": (demosaic_mlri2, reference_demosaic_mlri2),
    "ARI": (demosaic_ari, reference_demosaic_ari),
    "ARI2": (demosaic_ari2, reference_demosaic_ari2),
}


def _deterministic_rgb(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.int64)
    image = np.empty((*shape, 3), dtype=np.float64)
    image[:, :, 0] = (17 * rows + 31 * cols + (rows * cols) % 29) % 256
    image[:, :, 1] = (37 * rows + 11 * cols + np.bitwise_xor(rows, cols) * 3) % 256
    image[:, :, 2] = (13 * rows + 23 * cols + (rows - cols) ** 2) % 256
    return image


def _gradient_rgb(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float64)
    denom_r = max(shape[0] - 1, 1)
    denom_c = max(shape[1] - 1, 1)
    peak = 64.0
    image = np.empty((*shape, 3), dtype=np.float64)
    image[:, :, 0] = cols * peak / denom_c
    image[:, :, 1] = rows * peak / denom_r
    image[:, :, 2] = (rows / denom_r + cols / denom_c) * peak / 2.0
    return image


@pytest.mark.parametrize("package", (demosaic, demosaic_reference))
def test_public_package_api_matches(package):
    assert set(package.__all__) == set(demosaic.__all__)


def test_public_algorithm_api_matches():
    assert set(demosaic_reference.algorithms.__all__) == set(demosaic.algorithms.__all__)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("COLOR_BayerRGGB2BGR_RI", ("rggb", "RI")),
        ("COLOR_BayerGRBG2BGR_MLRI", ("grbg", "MLRI")),
        ("COLOR_BayerGBRG2BGR_MLRI2", ("gbrg", "MLRI2")),
        ("COLOR_BayerBGGR2BGR_ARI", ("bggr", "ARI")),
        ("COLOR_BayerBGGR2BGR_ARI2", ("bggr", "ARI2")),
    ],
)
def test_reference_parse_code_matches_current(code: str, expected: tuple[str, str]):
    assert tuple(parse_code(code)) == expected
    assert tuple(reference_parse_code(code)) == expected


@pytest.mark.parametrize("pattern", PATTERNS)
def test_reference_bayer_helpers_match_current(pattern: str):
    rgb = _deterministic_rgb((5, 6))

    np.testing.assert_array_equal(bayer_mask(rgb.shape[:2], pattern), reference_bayer_mask(rgb.shape[:2], pattern))

    mosaic, mask = mosaic_bayer(rgb, pattern)
    reference_mosaic, reference_mask = reference_mosaic_bayer(rgb, pattern)
    np.testing.assert_array_equal(mosaic, reference_mosaic)
    np.testing.assert_array_equal(mask, reference_mask)

    mask_gr, mask_gb = mask_gr_gb(rgb.shape[:2], pattern)
    reference_mask_gr, reference_mask_gb = reference_mask_gr_gb(rgb.shape[:2], pattern)
    np.testing.assert_array_equal(mask_gr, reference_mask_gr)
    np.testing.assert_array_equal(mask_gb, reference_mask_gb)


@pytest.mark.parametrize("rgb", (_deterministic_rgb(), _gradient_rgb()))
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("algorithm", tuple(ALGORITHMS))
def test_reference_algorithm_outputs_match_current_with_tolerance(
    rgb: np.ndarray,
    pattern: str,
    algorithm: str,
):
    current_algorithm, reference_algorithm = ALGORITHMS[algorithm]
    mosaic, mask = mosaic_bayer(rgb, pattern)
    reference_mosaic, reference_mask = reference_mosaic_bayer(rgb, pattern)

    output = current_algorithm(mosaic, mask, pattern)
    reference_output = reference_algorithm(reference_mosaic, reference_mask, pattern)

    assert output.shape == reference_output.shape == rgb.shape
    np.testing.assert_allclose(output, reference_output, rtol=0, atol=0.25)
