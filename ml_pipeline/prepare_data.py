# prepare_data.py
# Splits raw maize leaf images into train / val / test subsets.
#
# Usage:
#   python ml_pipeline/prepare_data.py
#
# Input:  ml_pipeline/data/raw/<class>/   (images you placed there)
# Output: ml_pipeline/dataset/train|val|test/<class>/

import os
import sys
import random
import shutil
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────────

# Paths are relative to the project root (maize-ai/), not this file's location.
# Run the script from the project root: python ml_pipeline/prepare_data.py
RAW_DIR     = Path("ml_pipeline/data/raw")
DATASET_DIR = Path("ml_pipeline/dataset")

# The four disease classes — folder names must match exactly
CLASSES = ["healthy", "common_rust", "leaf_blight", "gray_leaf_spot"]

# Valid image extensions — stored lowercase; we normalise every suffix with
# .lower() at scan time, so .JPG, .JPEG, .PNG, .WEBP are all matched.
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Split ratios — must sum to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Fixed seed so every run produces the same split (important for reproducibility)
RANDOM_SEED = 42

# Splits we will create
SPLITS = ["train", "val", "test"]


# ─── Helper: collect image paths from one class folder ─────────────────────────

def collect_images(class_dir: Path) -> list[Path]:
    """Return a sorted list of all valid image files inside class_dir.

    Uses rglob("*") to search recursively, so images that Kaggle unzipped into
    a nested subfolder (e.g. healthy/Healthy/*.jpg) are found automatically.
    Hidden files like .DS_Store and .gitkeep are skipped by the extension check.
    Extension matching is case-insensitive: .JPG and .jpg both match.
    """
    images = [
        p for p in class_dir.rglob("*")          # recursive — finds files in any subfolder
        if p.is_file()
        and p.suffix.lower() in VALID_EXTENSIONS  # .lower() handles .JPG, .PNG, etc.
    ]
    # Sort so the order is deterministic before we shuffle
    return sorted(images)


# ─── Helper: split a list into three parts by ratio ────────────────────────────

def split_files(files: list, train_r: float, val_r: float) -> tuple:
    """Return (train_files, val_files, test_files) from a flat list.

    Slices the list at the computed index boundaries.
    test gets whatever is left after train and val, so the ratios are exact.
    """
    n = len(files)
    train_end = int(n * train_r)          # index where train portion ends
    val_end   = train_end + int(n * val_r) # index where val portion ends

    return files[:train_end], files[train_end:val_end], files[val_end:]


# ─── Helper: copy a list of files into a destination folder ────────────────────

def copy_files(files: list[Path], dest_dir: Path) -> None:
    """Copy each file in files into dest_dir, preserving the filename."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir / src.name)  # copy2 preserves file metadata


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── 1. Verify we're being run from the project root ──────────────────────
    if not RAW_DIR.exists():
        print(
            "\nError: could not find ml_pipeline/data/raw/\n"
            "Make sure you are running this script from the project root:\n"
            "  cd maize-ai\n"
            "  python ml_pipeline/prepare_data.py\n"
        )
        sys.exit(1)

    # ── 2. Check each class folder for images; collect problems ──────────────
    print("\nScanning raw image folders...")

    class_images: dict[str, list[Path]] = {}  # class name → list of image paths
    problems: list[str] = []                   # list of human-readable problem messages

    for cls in CLASSES:
        cls_dir = RAW_DIR / cls

        if not cls_dir.exists():
            # The folder itself is missing
            problems.append(
                f"  [{cls}]  folder not found — expected: {cls_dir}"
            )
            continue

        images = collect_images(cls_dir)

        if len(images) == 0:
            # Folder exists but is empty (or contains no valid images)
            problems.append(
                f"  [{cls}]  folder is empty — place .jpg / .png images inside {cls_dir}"
            )
            continue

        if len(images) < 10:
            # Too few images to split meaningfully — warn but don't stop
            print(
                f"  Warning: [{cls}] only has {len(images)} image(s). "
                "Splits will be very small. At least ~30 images per class is recommended."
            )

        class_images[cls] = images

        # Show total count and a few example filenames so you can confirm
        # the script is reading the right folder (and not some hidden cache)
        print(f"  [{cls}]  {len(images)} images found")
        sample = images[:3]   # first 3 after sorting (before shuffle)
        for p in sample:
            # Print relative path from RAW_DIR so nested subfolders are visible
            print(f"           {p.relative_to(RAW_DIR)}")

    # ── 3. Stop if any class has a problem ───────────────────────────────────
    if problems:
        print("\n" + "=" * 60)
        print("Cannot continue — some class folders need attention:\n")
        for msg in problems:
            print(msg)
        print(
            "\nWhat to do:\n"
            "  1. Download the dataset from Kaggle:\n"
            "     https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset\n"
            "  2. Unzip it and copy images into the matching folders above.\n"
            "  3. Re-run:  python ml_pipeline/prepare_data.py\n"
        )
        print("=" * 60 + "\n")
        sys.exit(1)

    # ── 4. Clear the existing dataset/ folder so we start clean ──────────────
    if DATASET_DIR.exists():
        print(f"\nRemoving existing dataset at {DATASET_DIR} ...")
        shutil.rmtree(DATASET_DIR)   # delete the whole tree
    DATASET_DIR.mkdir(parents=True)
    print(f"Created fresh dataset directory: {DATASET_DIR}")

    # ── 5. Shuffle and split each class independently ─────────────────────────
    print(f"\nSplitting with seed={RANDOM_SEED}  "
          f"(train {int(TRAIN_RATIO*100)}% / val {int(VAL_RATIO*100)}% / test {int(TEST_RATIO*100)}%)")

    # This dict collects counts for the summary table:  counts[split][class] = n
    counts: dict[str, dict[str, int]] = {s: {} for s in SPLITS}

    rng = random.Random(RANDOM_SEED)   # isolated RNG so we don't affect global state

    for cls, images in class_images.items():
        # Shuffle in place using our seeded RNG
        shuffled = images[:]          # make a copy so we don't mutate the original list
        rng.shuffle(shuffled)

        # Split into three portions
        train_files, val_files, test_files = split_files(shuffled, TRAIN_RATIO, VAL_RATIO)

        # Copy each portion into the correct destination subfolder
        copy_files(train_files, DATASET_DIR / "train" / cls)
        copy_files(val_files,   DATASET_DIR / "val"   / cls)
        copy_files(test_files,  DATASET_DIR / "test"  / cls)

        # Record counts for the summary table
        counts["train"][cls] = len(train_files)
        counts["val"][cls]   = len(val_files)
        counts["test"][cls]  = len(test_files)

    # ── 6. Print summary table ────────────────────────────────────────────────
    col_w = 16   # width of each column in the table

    print("\n" + "=" * 60)
    print("Dataset split complete — image counts per class per split:")
    print("=" * 60)

    # Header row
    header = f"{'Class':<20}" + "".join(f"{s:>{col_w}}" for s in SPLITS) + f"{'Total':>{col_w}}"
    print(header)
    print("-" * len(header))

    # One row per class
    totals = {s: 0 for s in SPLITS}
    for cls in CLASSES:
        row_counts = [counts[s][cls] for s in SPLITS]
        row_total  = sum(row_counts)
        row = f"{cls:<20}" + "".join(f"{c:>{col_w}}" for c in row_counts) + f"{row_total:>{col_w}}"
        print(row)
        for s, c in zip(SPLITS, row_counts):
            totals[s] += c

    # Totals footer
    print("-" * len(header))
    grand_total = sum(totals.values())
    footer = f"{'TOTAL':<20}" + "".join(f"{totals[s]:>{col_w}}" for s in SPLITS) + f"{grand_total:>{col_w}}"
    print(footer)
    print("=" * 60)

    print(f"\nAll images copied to: {DATASET_DIR.resolve()}")
    print("Raw images in data/raw/ are unchanged.\n")
    print("Next step:  python ml_pipeline/train.py\n")


if __name__ == "__main__":
    main()
