from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from demosaic import cpsnr, demosaic, mosaicing_cfa_bayer, psnr, ssim

TARGET_ALGORITHMS = ("RI", "MLRI2", "ARI", "ARI2")
ALL_ALGORITHMS = ("RI", "MLRI", "MLRI2", "ARI", "ARI2")
DATASETS = {
    "IMAX": [Path("datasets/IMAX") / f"{i}.tif" for i in range(1, 19)],
    "Kodak": [Path("datasets/Kodak") / f"img{i}.bmp" for i in range(1, 13)],
}


def _code(algorithm: str, pattern: str = "GRBG") -> str:
    return f"COLOR_Bayer{pattern.upper()}2BGR_{algorithm}"


def _evaluate_image(path: Path, algorithm: str) -> tuple[np.ndarray, float, np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    cfa = mosaicing_cfa_bayer(bgr, "GRBG")
    out_bgr = demosaic(cfa, _code(algorithm))
    ref_rgb = bgr[:, :, ::-1].astype(np.float64)
    out_rgb = out_bgr[:, :, ::-1]
    return psnr(ref_rgb, out_rgb, peak=255, border=10), cpsnr(ref_rgb, out_rgb, peak=255, border=10), ssim(ref_rgb, out_rgb)


def _dataset_paths(dataset: str, limit: int | None) -> list[Path]:
    paths = DATASETS[dataset]
    if limit is not None:
        paths = paths[:limit]
    return paths


def run_dataset(dataset: str, algorithm: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    psnr_rows: list[np.ndarray] = []
    ssim_rows: list[np.ndarray] = []
    for path in _dataset_paths(dataset, limit):
        channel_psnr, combined, channel_ssim = _evaluate_image(path, algorithm)
        psnr_rows.append(np.append(channel_psnr, combined))
        ssim_rows.append(channel_ssim)
        print(f"{algorithm} {dataset} {path.name}: PSNR/CPSNR {psnr_rows[-1]} SSIM {channel_ssim}", flush=True)
    return np.vstack(psnr_rows), np.vstack(ssim_rows)


def _format_row(name: str, imax: np.ndarray, kodak: np.ndarray, combined: np.ndarray, ssim_table: bool = False) -> str:
    if ssim_table:
        return f"| {name:<9} | {imax[0]:.4f} | {imax[1]:.4f} | {imax[2]:.4f} | {imax[3]:.4f} | " f"{kodak[0]:.4f} | {kodak[1]:.4f} | {kodak[2]:.4f} | {kodak[3]:.4f} | " f"{combined[0]:.4f} | {combined[1]:.4f} | {combined[2]:.4f} | {combined[3]:.4f} |"
    return f"| {name:<9} | {imax[0]:.2f} | {imax[1]:.2f} | {imax[2]:.2f} | {imax[3]:.2f} | " f"{kodak[0]:.2f} | {kodak[1]:.2f} | {kodak[2]:.2f} | {kodak[3]:.2f} | " f"{combined[0]:.2f} | {combined[1]:.2f} | {combined[2]:.2f} | {combined[3]:.2f} |"


def run_benchmark(limit: int | None = None, algorithms: tuple[str, ...] = TARGET_ALGORITHMS) -> None:
    psnr_summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ssim_summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for algorithm in algorithms:
        imax_psnr, imax_ssim = run_dataset("IMAX", algorithm, limit)
        kodak_psnr, kodak_ssim = run_dataset("Kodak", algorithm, limit)
        all_psnr = np.vstack([imax_psnr, kodak_psnr])
        all_ssim = np.vstack([imax_ssim, kodak_ssim])
        psnr_summary[algorithm] = (imax_psnr.mean(axis=0), kodak_psnr.mean(axis=0), all_psnr.mean(axis=0))
        ssim_summary[algorithm] = (imax_ssim.mean(axis=0), kodak_ssim.mean(axis=0), all_ssim.mean(axis=0))

    print("\nPSNR/CPSNR")
    for algorithm, (imax, kodak, combined) in psnr_summary.items():
        print(_format_row(algorithm, imax, kodak, combined))

    print("\nSSIM")
    for algorithm, (imax, kodak, combined) in ssim_summary.items():
        print(_format_row(algorithm, imax, kodak, combined, ssim_table=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="run target IMAX/Kodak benchmark")
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="run one dataset")
    parser.add_argument("--algorithm", choices=ALL_ALGORITHMS, default="RI")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark:
        run_benchmark(limit=args.limit)
    elif args.dataset:
        rows, ssim_rows = run_dataset(args.dataset, args.algorithm, args.limit)
        print("PSNR/CPSNR mean:", rows.mean(axis=0))
        print("SSIM mean:", ssim_rows.mean(axis=0))
    else:
        raise SystemExit("Use --benchmark or --dataset {IMAX,Kodak}.")


if __name__ == "__main__":
    main()
