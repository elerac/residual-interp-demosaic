from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from demosaic import demosaic

METHODS = ("RI", "MLRI", "MLRI2", "ARI", "ARI2")


def run(input_path: Path, output_dir: Path, pattern: str = "GRBG") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    stem = input_path.stem

    for algorithm in METHODS:
        result = demosaic(image, f"COLOR_Bayer{pattern.upper()}2BGR_{algorithm}")
        output_path = output_dir / f"{stem}_{algorithm}.png"
        if not cv2.imwrite(str(output_path), result.astype("uint8")):
            raise RuntimeError(f"failed to write {output_path}")
        print(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply several demosaicing methods to one image.")
    parser.add_argument("input", nargs="?", default="tshirts.jpg", help="input image path")
    parser.add_argument("--output-dir", default="results/demosaic", help="directory for exported images")
    parser.add_argument("--pattern", default="GRBG", choices=("RGGB", "GRBG", "GBRG", "BGGR"), help="Bayer pattern code")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.input), Path(args.output_dir), pattern=args.pattern)


if __name__ == "__main__":
    main()
