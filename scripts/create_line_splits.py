from pathlib import Path
import random


SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_GT = Path("/home/edxnG08/projecte-deep-learning-08/grup_8/iam_dataset/linux_gt.txt")
OUTPUT_DIR = PROJECT_ROOT / "splits"


def main():
    lines = [line for line in SOURCE_GT.read_text().splitlines() if line.strip()]
    random.Random(SEED).shuffle(lines)

    train_end = int(len(lines) * TRAIN_RATIO)
    val_end = train_end + int(len(lines) * VAL_RATIO)

    splits = {
        "train_gt_80.txt": lines[:train_end],
        "val_gt_10.txt": lines[train_end:val_end],
        "test_gt_10.txt": lines[val_end:],
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, split_lines in splits.items():
        (OUTPUT_DIR / filename).write_text("\n".join(split_lines) + "\n")
        print(f"{filename}: {len(split_lines)} muestras")


if __name__ == "__main__":
    main()
