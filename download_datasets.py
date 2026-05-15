from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BENCHMARK_URL = "http://www.ok.sc.e.titech.ac.jp/res/DM/Benchmark.zip"
ARCHIVE_DATASET_PREFIX = "Benchmark/datasets"
EXPECTED_FILES = {
    "IMAX": tuple(f"{i}.tif" for i in range(1, 19)),
    "Kodak": tuple(f"img{i}.bmp" for i in range(1, 13)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the IMAX/Kodak benchmark datasets used by benchmark.py."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets"),
        help="dataset output directory (default: datasets)",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="download again and overwrite existing dataset files",
    )
    return parser.parse_args()


def download_archive(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_name(f"{archive_path.name}.download")
    print(f"Downloading: {url}")
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as target:
            shutil.copyfileobj(response, target)
        tmp_path.replace(archive_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def expected_archive_members() -> dict[str, Path]:
    members: dict[str, Path] = {}
    for dataset, filenames in EXPECTED_FILES.items():
        for filename in filenames:
            archive_name = f"{ARCHIVE_DATASET_PREFIX}/{dataset}/{filename}"
            members[archive_name] = Path(dataset) / filename
    return members


def safe_destination(root: Path, relative_path: Path) -> Path:
    root_resolved = root.resolve()
    destination = (root / relative_path).resolve()
    try:
        destination.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to extract outside {root}: {relative_path}") from exc
    return destination


def extract_datasets(archive_path: Path, root: Path, force: bool) -> tuple[int, int]:
    expected_members = expected_archive_members()
    extracted = 0
    skipped = 0

    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive_members = set(archive.namelist())
        missing = sorted(set(expected_members) - archive_members)
        if missing:
            missing_list = "\n".join(f"  {name}" for name in missing)
            raise RuntimeError(f"Archive is missing expected dataset files:\n{missing_list}")

        for archive_name, relative_path in expected_members.items():
            info = archive.getinfo(archive_name)
            if info.is_dir():
                raise RuntimeError(f"Expected file but found directory: {archive_name}")

            destination = safe_destination(root, relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not force:
                skipped += 1
                continue

            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1

    return extracted, skipped


def dataset_errors(root: Path) -> list[str]:
    errors: list[str] = []
    for dataset, filenames in EXPECTED_FILES.items():
        dataset_dir = root / dataset
        expected = set(filenames)
        if dataset_dir.exists() and not dataset_dir.is_dir():
            errors.append(f"{dataset}: expected directory at {dataset_dir}")
            actual = set()
        elif dataset_dir.exists():
            actual = {path.name for path in dataset_dir.iterdir() if path.is_file()}
        else:
            actual = set()
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"{dataset}: missing {', '.join(missing)}")
        if extra:
            errors.append(f"{dataset}: unexpected {', '.join(extra)}")
    return errors


def verify_datasets(root: Path) -> None:
    errors = dataset_errors(root)
    if errors:
        raise RuntimeError("Dataset verification failed:\n" + "\n".join(f"  {error}" for error in errors))


def main() -> int:
    args = parse_args()
    total = sum(len(files) for files in EXPECTED_FILES.values())

    print(f"Source URL: {BENCHMARK_URL}")
    print(f"Dataset root: {args.root}")

    if not args.force_extract and not dataset_errors(args.root):
        print("Datasets already complete; skipping download.")
        print(f"Verified files: {total}")
        return 0

    with tempfile.TemporaryDirectory(prefix="ri-benchmark-") as tmpdir:
        archive_path = Path(tmpdir) / "Benchmark.zip"
        download_archive(BENCHMARK_URL, archive_path)
        extracted, skipped = extract_datasets(archive_path, args.root, args.force_extract)

    verify_datasets(args.root)

    print(f"Extracted files: {extracted}")
    print(f"Skipped existing files: {skipped}")
    print(f"Verified files: {total}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
