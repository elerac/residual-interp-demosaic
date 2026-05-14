from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


PATTERNS = {"rggb", "grbg", "gbrg", "bggr"}
ALGORITHMS = {"RI", "MLRI", "MLRI2", "ARI", "ARI2"}

_CODE_RE = re.compile(
    r"^COLOR_Bayer(?P<pattern>RGGB|GRBG|GBRG|BGGR)2BGR_(?P<algorithm>RI|MLRI|MLRI2|ARI|ARI2)$"
)


@dataclass(frozen=True)
class DemosaicCode:
    pattern: str
    algorithm: str

    def __iter__(self):
        yield self.pattern
        yield self.algorithm

    def __getitem__(self, index: int) -> str:
        return (self.pattern, self.algorithm)[index]

    def __len__(self) -> int:
        return 2

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return (self.pattern, self.algorithm) == other
        if isinstance(other, DemosaicCode):
            return (self.pattern, self.algorithm) == (other.pattern, other.algorithm)
        return NotImplemented


def parse_code(code: str) -> DemosaicCode:
    match = _CODE_RE.match(code)
    if not match:
        raise ValueError(
            "code must look like COLOR_BayerGRBG2BGR_RI, with pattern "
            "RGGB/GRBG/GBRG/BGGR and algorithm RI/MLRI/MLRI2/ARI/ARI2"
        )
    return DemosaicCode(
        pattern=match.group("pattern").lower(),
        algorithm=match.group("algorithm"),
    )


def _pattern_channels(pattern: str) -> list[int]:
    pattern = pattern.lower()
    if pattern not in PATTERNS:
        raise ValueError(f"unsupported Bayer pattern: {pattern!r}")
    channel = {"r": 0, "g": 1, "b": 2}
    return [channel[c] for c in pattern]


def bayer_mask(shape: tuple[int, int], pattern: str) -> np.ndarray:
    height, width = shape
    channels = _pattern_channels(pattern)
    mask = np.zeros((height, width, 3), dtype=np.float64)
    rows1 = slice(0, None, 2)
    rows2 = slice(1, None, 2)
    cols1 = slice(0, None, 2)
    cols2 = slice(1, None, 2)
    mask[rows1, cols1, channels[0]] = 1.0
    mask[rows1, cols2, channels[1]] = 1.0
    mask[rows2, cols1, channels[2]] = 1.0
    mask[rows2, cols2, channels[3]] = 1.0
    return mask


def mosaic_bayer(rgb: np.ndarray, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    mask = bayer_mask(rgb.shape[:2], pattern)
    return rgb * mask, mask


def mask_gr_gb(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, np.ndarray]:
    pattern = pattern.lower()
    if pattern not in PATTERNS:
        raise ValueError(f"unsupported Bayer pattern: {pattern!r}")
    height, width = shape
    mask_gr = np.zeros((height, width), dtype=np.float64)
    mask_gb = np.zeros((height, width), dtype=np.float64)
    if pattern == "grbg":
        mask_gr[0::2, 0::2] = 1.0
        mask_gb[1::2, 1::2] = 1.0
    elif pattern == "rggb":
        mask_gr[0::2, 1::2] = 1.0
        mask_gb[1::2, 0::2] = 1.0
    elif pattern == "gbrg":
        mask_gb[0::2, 0::2] = 1.0
        mask_gr[1::2, 1::2] = 1.0
    elif pattern == "bggr":
        mask_gb[0::2, 1::2] = 1.0
        mask_gr[1::2, 0::2] = 1.0
    return mask_gr, mask_gb
