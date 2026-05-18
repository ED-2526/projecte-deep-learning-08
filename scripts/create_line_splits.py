from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Optional


SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUP_ROOT = PROJECT_ROOT.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "splits"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    default_root: Path
    source_gt: Optional[str] = None
    parquet_files: tuple[str, ...] = ()


DATASETS = {
    "iam": DatasetConfig(
        name="iam",
        default_root=GROUP_ROOT / "iam_dataset",
        source_gt="linux_gt.txt",
    ),
    "esposalles": DatasetConfig(
        name="esposalles",
        default_root=Path("/home/datasets/esposalles"),
        parquet_files=("train.parquet", "validation.parquet", "test.parquet"),
    ),
}


def read_gt_file(gt_path):
    return [line for line in gt_path.read_text().splitlines() if line.strip()]


def image_extension(image_bytes):
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return ".tiff"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def read_parquet_dataset(dataset_root, image_root, parquet_files, extract_images=True):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "Error: para leer Esposalles en formato parquet necesitas instalar pyarrow "
            "(por ejemplo: pip install pyarrow).",
            file=sys.stderr,
        )
        raise

    lines = []
    image_dir = image_root / "images"
    if extract_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    for parquet_name in parquet_files:
        parquet_path = dataset_root / parquet_name
        if not parquet_path.exists():
            raise FileNotFoundError(f"No se encuentra {parquet_path}")

        columns = ["text", "image"] if extract_images else ["text"]
        table = pq.read_table(parquet_path, columns=columns)
        texts = table.column("text").to_pylist()
        images = table.column("image").to_pylist() if extract_images else [None] * len(texts)

        for idx, (text, image) in enumerate(zip(texts, images)):
            if text is None or not str(text).strip():
                continue

            if extract_images:
                image_bytes = image.get("bytes") if isinstance(image, dict) else None
                if not image_bytes:
                    continue

                extension = image_extension(image_bytes)
                rel_path = Path("images") / f"{parquet_path.stem}_{idx:06d}{extension}"
                out_path = image_root / rel_path
                if not out_path.exists():
                    out_path.write_bytes(image_bytes)
            else:
                rel_path = Path(f"{parquet_name}:{idx}")

            lines.append(f"{rel_path.as_posix()}\t{str(text).strip()}")

    return lines


def load_dataset_lines(dataset_name, dataset_root, image_root, extract_images=True):
    config = DATASETS[dataset_name]
    if config.source_gt is not None:
        gt_path = dataset_root / config.source_gt
        if not gt_path.exists():
            raise FileNotFoundError(f"No se encuentra {gt_path}")
        return read_gt_file(gt_path), dataset_root

    return read_parquet_dataset(
        dataset_root,
        image_root,
        config.parquet_files,
        extract_images,
    ), image_root


def split_lines(lines, seed=SEED, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    lines = list(lines)
    random.Random(seed).shuffle(lines)

    train_end = int(len(lines) * train_ratio)
    val_end = train_end + int(len(lines) * val_ratio)

    return {
        "train_gt_80.txt": lines[:train_end],
        "val_gt_10.txt": lines[train_end:val_end],
        "test_gt_10.txt": lines[val_end:],
    }


def write_splits(splits, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_lines in splits.items():
        (output_dir / filename).write_text("\n".join(split_lines) + "\n")
        print(f"{filename}: {len(split_lines)} muestras")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera splits 80/10/10 para IAM o Esposalles."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=DATASETS.keys(),
        default="iam",
        help="Dataset a procesar.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Ruta al directorio del dataset. Si no se indica, se usa la ruta por defecto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio donde guardar los txt de split.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Directorio base donde extraer las imagenes de datasets parquet.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-extract-images",
        action="store_true",
        help="Para datasets parquet, genera referencias parquet:fila sin extraer imagenes.",
    )
    return parser.parse_args()


def default_output_dir(dataset_name):
    if dataset_name == "iam":
        return OUTPUT_DIR
    return OUTPUT_DIR / dataset_name


def main():
    args = parse_args()
    dataset_root = args.dataset_root or DATASETS[args.dataset].default_root
    output_dir = args.output_dir or default_output_dir(args.dataset)
    image_root = args.image_root or (PROJECT_ROOT / "data" / args.dataset)

    lines, img_dir = load_dataset_lines(
        args.dataset,
        dataset_root,
        image_root,
        extract_images=not args.no_extract_images,
    )
    splits = split_lines(lines, seed=args.seed)
    write_splits(splits, output_dir)

    print(f"Splits guardados en: {output_dir}")
    print(f"img_dir para este dataset: {img_dir}")


if __name__ == "__main__":
    main()
