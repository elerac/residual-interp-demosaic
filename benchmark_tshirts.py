from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004, demosaicing_CFA_Bayer_Menon2007

from demosaic import bayer_mask, cpsnr, demosaic, mosaicing_cfa_bayer, ssim


PATTERNS = ("RGGB", "GRBG", "GBRG", "BGGR")
DEFAULT_CROPS = {
    "crop1": (530, 318),
    "crop2": (382, 555),
}

OPENCV_BILINEAR_CODES = {
    "RGGB": cv2.COLOR_BayerRGGB2BGR,
    "GRBG": cv2.COLOR_BayerGRBG2BGR,
    "GBRG": cv2.COLOR_BayerGBRG2BGR,
    "BGGR": cv2.COLOR_BayerBGGR2BGR,
}

OPENCV_EA_CODES = {
    "RGGB": cv2.COLOR_BayerRGGB2BGR_EA,
    "GRBG": cv2.COLOR_BayerGRBG2BGR_EA,
    "GBRG": cv2.COLOR_BayerGBRG2BGR_EA,
    "BGGR": cv2.COLOR_BayerBGGR2BGR_EA,
}


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    implementation: str
    run: Callable[[], np.ndarray]


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def _as_bgr_uint8_from_rgb01(image_rgb: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.rint(np.clip(image_rgb, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)
    return clipped[:, :, ::-1]


def _crop_and_zoom(image: np.ndarray, x0: int, y0: int, width: int, height: int, scale: int) -> np.ndarray:
    crop = image[y0 : y0 + height, x0 : x0 + width]
    return cv2.resize(crop, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)


def _colored_cfa(cfa: np.ndarray, mask_rgb: np.ndarray) -> np.ndarray:
    cfa_rgb = np.zeros((*cfa.shape, 3), dtype=np.uint8)
    for channel in range(3):
        channel_mask = mask_rgb[:, :, channel].astype(bool)
        cfa_rgb[:, :, channel][channel_mask] = cfa[channel_mask]
    return cfa_rgb[:, :, ::-1]


def _safe_crop_specs(
    image_shape: tuple[int, int],
    crop_specs: dict[str, tuple[int, int]],
    crop_size: int,
) -> dict[str, tuple[int, int]]:
    height, width = image_shape
    if crop_size <= 0:
        raise ValueError("crop size must be positive")
    if crop_size > height or crop_size > width:
        raise ValueError("crop size must fit inside the input image")
    return {
        name: (
            min(max(0, x0), width - crop_size),
            min(max(0, y0), height - crop_size),
        )
        for name, (x0, y0) in crop_specs.items()
    }


def _time_method(method: Method, runs: int) -> tuple[np.ndarray, list[float]]:
    if runs <= 0:
        raise ValueError("runs must be positive")
    method.run()
    times: list[float] = []
    output = None
    for _ in range(runs):
        start = time.perf_counter()
        output = method.run()
        times.append(time.perf_counter() - start)
    if output is None:
        raise RuntimeError(f"{method.key} did not produce output")
    return output, times


def _compute_metrics(reference_rgb: np.ndarray, output_bgr: np.ndarray) -> dict[str, Any]:
    output_rgb = output_bgr[:, :, ::-1].astype(np.float64)
    ssim_values = ssim(reference_rgb, output_rgb)
    return {
        "cpsnr_db": cpsnr(reference_rgb, output_rgb, peak=255, border=0),
        "ssim": {
            "r": float(ssim_values[0]),
            "g": float(ssim_values[1]),
            "b": float(ssim_values[2]),
            "avg": float(ssim_values[3]),
        },
    }


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _markdown_summary(
    *,
    image_path: Path,
    image_shape: tuple[int, int],
    pattern: str,
    runs: int,
    output_dir: Path,
    results: list[dict[str, Any]],
) -> str:
    stem = image_path.stem
    width, height = image_shape[1], image_shape[0]
    rows = [
        "| Method | Implementation | CPSNR (dB) | SSIM Avg | Time (s) [mean +/- std] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for result in sorted(results, key=lambda item: float(item["cpsnr_db"]), reverse=True):
        rows.append(
            "| {method} | {implementation} | {cpsnr:.2f} | {ssim_avg:.4f} | {mean:.4f}+/-{std:.4f} |".format(
                method=result["method"],
                implementation=result["implementation"],
                cpsnr=float(result["cpsnr_db"]),
                ssim_avg=float(result["ssim"]["avg"]),
                mean=float(result["time_mean_s"]),
                std=float(result["time_std_s"]),
            )
        )

    artifact_prefix = output_dir.as_posix()
    return "\n".join(
        [
            f"Benchmark image: `{image_path}` ({width} x {height}), Bayer pattern `{pattern}`, {runs} timed runs after one warmup.",
            "",
            *rows,
            "",
            "### Input and CFA",
            "",
            "| Input (BGR) | Bayer CFA (RGB-colored) |",
            "| --- | --- |",
            f"| ![input]({artifact_prefix}/{stem}_input.png) | ![cfa]({artifact_prefix}/{stem}_cfa_rgb.png) |",
            "",
            "### CFA Cropped Images (4x nearest-neighbor zoom)",
            "",
            "| Crop Region | Original Input | RGB-colored CFA |",
            "| --- | --- | --- |",
            f"| crop1 | ![input crop1]({artifact_prefix}/{stem}_input_crop1.png) | ![cfa crop1]({artifact_prefix}/{stem}_cfa_rgb_crop1.png) |",
            f"| crop2 | ![input crop2]({artifact_prefix}/{stem}_input_crop2.png) | ![cfa crop2]({artifact_prefix}/{stem}_cfa_rgb_crop2.png) |",
            "",
            "### Demosaiced Cropped Images (4x nearest-neighbor zoom)",
            "",
            "| Method | Crop1 | Crop2 |",
            "| :---: | :---: | :---: |",
            f"| Original | ![original crop1]({artifact_prefix}/{stem}_input_crop1.png) | ![original crop2]({artifact_prefix}/{stem}_input_crop2.png) |",
            *[
                "| {method} | ![{key} crop1]({prefix}/{stem}_demosaiced_{key}_crop1.png) | ![{key} crop2]({prefix}/{stem}_demosaiced_{key}_crop2.png) |".format(
                    method=result["method"],
                    key=result["key"],
                    prefix=artifact_prefix,
                    stem=stem,
                )
                for result in results
            ],
            "",
        ]
    )


def run_benchmark(input_path: Path, output_dir: Path, pattern: str, runs: int, crop_size: int, zoom: int) -> list[dict[str, Any]]:
    pattern = pattern.upper()
    if pattern not in PATTERNS:
        raise ValueError(f"unsupported Bayer pattern: {pattern}")
    if zoom <= 0:
        raise ValueError("zoom must be positive")

    image_bgr = _read_bgr(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    mask_rgb = bayer_mask(image_bgr.shape[:2], pattern.lower())
    cfa = mosaicing_cfa_bayer(image_bgr, pattern)
    cfa_float = cfa.astype(np.float32) / 255.0
    crop_specs = _safe_crop_specs(image_bgr.shape[:2], DEFAULT_CROPS, crop_size)

    _write_png(output_dir / f"{stem}_input.png", image_bgr)
    _write_png(output_dir / f"{stem}_cfa.png", cfa)
    cfa_bgr = _colored_cfa(cfa, mask_rgb)
    _write_png(output_dir / f"{stem}_cfa_rgb.png", cfa_bgr)
    for crop_name, (x0, y0) in crop_specs.items():
        _write_png(
            output_dir / f"{stem}_input_{crop_name}.png",
            _crop_and_zoom(image_bgr, x0, y0, crop_size, crop_size, zoom),
        )
        _write_png(
            output_dir / f"{stem}_cfa_rgb_{crop_name}.png",
            _crop_and_zoom(cfa_bgr, x0, y0, crop_size, crop_size, zoom),
        )

    local_methods: list[tuple[str, str]] = [
        ("ri", "RI"),
        ("mlri", "MLRI"),
        ("mlri2", "MLRI2"),
        ("ari", "ARI"),
        ("ari2", "ARI2"),
    ]
    methods = [
        Method(
            key=key,
            label=label,
            implementation="This repository",
            run=lambda label=label: np.clip(
                np.rint(demosaic(cfa, f"COLOR_Bayer{pattern}2BGR_{label}")), 0, 255
            ).astype(np.uint8),
        )
        for key, label in local_methods
    ]
    methods.extend(
        [
            Method(
                key="opencv_bilinear",
                label="Bilinear",
                implementation="OpenCV",
                run=lambda: cv2.demosaicing(cfa, OPENCV_BILINEAR_CODES[pattern]),
            ),
            Method(
                key="opencv_ea",
                label="Edge-Aware",
                implementation="OpenCV",
                run=lambda: cv2.demosaicing(cfa, OPENCV_EA_CODES[pattern]),
            ),
            Method(
                key="colour_malvar2004",
                label="Malvar2004",
                implementation="colour_demosaicing",
                run=lambda: _as_bgr_uint8_from_rgb01(demosaicing_CFA_Bayer_Malvar2004(cfa_float, pattern=pattern)),
            ),
            Method(
                key="colour_menon2007",
                label="Menon2007",
                implementation="colour_demosaicing",
                run=lambda: _as_bgr_uint8_from_rgb01(demosaicing_CFA_Bayer_Menon2007(cfa_float, pattern=pattern)),
            ),
        ]
    )

    reference_rgb = image_bgr[:, :, ::-1].astype(np.float64)
    results: list[dict[str, Any]] = []
    for method in methods:
        print(f"Processing {method.label} ({method.implementation})...", flush=True)
        output_bgr, times = _time_method(method, runs)
        output_path = output_dir / f"{stem}_demosaiced_{method.key}.png"
        _write_png(output_path, output_bgr)
        for crop_name, (x0, y0) in crop_specs.items():
            _write_png(
                output_dir / f"{stem}_demosaiced_{method.key}_{crop_name}.png",
                _crop_and_zoom(output_bgr, x0, y0, crop_size, crop_size, zoom),
            )
        result = {
            "key": method.key,
            "method": method.label,
            "implementation": method.implementation,
            **_compute_metrics(reference_rgb, output_bgr),
            "time_mean_s": float(np.mean(times)),
            "time_std_s": float(np.std(times)),
            "runs": runs,
            "output": output_path.as_posix(),
        }
        results.append(result)
        print(
            "{method:14s} | CPSNR: {cpsnr:8.4f} dB | SSIM: {ssim_avg:.4f} | Time: {mean:8.4f}+/-{std:7.4f} s".format(
                method=method.label,
                cpsnr=result["cpsnr_db"],
                ssim_avg=result["ssim"]["avg"],
                mean=result["time_mean_s"],
                std=result["time_std_s"],
            ),
            flush=True,
        )

    metadata = {
        "input": input_path.as_posix(),
        "image_shape": list(image_bgr.shape),
        "pattern": pattern,
        "runs": runs,
        "warmup_runs": 1,
        "crop_size": crop_size,
        "zoom": zoom,
        "crops": {name: [x0, y0] for name, (x0, y0) in crop_specs.items()},
        "results": results,
    }
    (output_dir / "benchmark_results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "benchmark_summary.md").write_text(
        _markdown_summary(
            image_path=input_path,
            image_shape=image_bgr.shape[:2],
            pattern=pattern,
            runs=runs,
            output_dir=output_dir,
            results=results,
        ),
        encoding="utf-8",
    )
    return results


def _resolve_output_path(output_dir: Path, output: str) -> Path:
    path = Path(output)
    if path.exists() or path.is_absolute():
        return path
    fallback = output_dir / path.name
    if fallback.exists():
        return fallback
    return path


def run_metrics_only(output_dir: Path) -> list[dict[str, Any]]:
    results_path = output_dir / "benchmark_results.json"
    metadata = json.loads(results_path.read_text(encoding="utf-8"))
    input_path = Path(metadata["input"])
    image_bgr = _read_bgr(input_path)
    reference_rgb = image_bgr[:, :, ::-1].astype(np.float64)

    results = metadata["results"]
    for result in results:
        output_path = _resolve_output_path(output_dir, result["output"])
        output_bgr = _read_bgr(output_path)
        result.update(_compute_metrics(reference_rgb, output_bgr))
        print(
            "{method:14s} | CPSNR: {cpsnr:8.4f} dB | SSIM: {ssim_avg:.4f} | preserved time: {mean:8.4f}+/-{std:7.4f} s".format(
                method=result["method"],
                cpsnr=result["cpsnr_db"],
                ssim_avg=result["ssim"]["avg"],
                mean=result["time_mean_s"],
                std=result["time_std_s"],
            ),
            flush=True,
        )

    results_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "benchmark_summary.md").write_text(
        _markdown_summary(
            image_path=input_path,
            image_shape=tuple(metadata["image_shape"][:2]),
            pattern=metadata["pattern"],
            runs=metadata["runs"],
            output_dir=output_dir,
            results=results,
        ),
        encoding="utf-8",
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark demosaicing methods on tshirts.jpg.")
    parser.add_argument("--input", type=Path, default=Path("tshirts.jpg"), help="input image path")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tshirts"), help="artifact output directory")
    parser.add_argument("--pattern", default="RGGB", choices=PATTERNS, help="Bayer pattern")
    parser.add_argument("--runs", type=int, default=5, help="timed runs per method after one warmup")
    parser.add_argument("--crop-size", type=int, default=100, help="square crop size")
    parser.add_argument("--zoom", type=int, default=4, help="nearest-neighbor crop zoom")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="recompute CPSNR/SSIM from existing output images without rerunning demosaicing or timing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metrics_only:
        run_metrics_only(args.output_dir)
    else:
        run_benchmark(args.input, args.output_dir, args.pattern, args.runs, args.crop_size, args.zoom)


if __name__ == "__main__":
    main()
