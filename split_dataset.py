"""Safe dataset splitting script for YOLO format

Features:
- Copy (default) or move image/label pairs into train/val splits
- Dry-run mode to preview actions without writing files
- Backup source dataset before moving (optional)
- Overwrite protection for output directory (requires explicit flag)
- Optionally generate `names` section from a `category_counts.json` file
"""
from pathlib import Path
import argparse
import random
import shutil
import json
import sys
import datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split YOLO dataset safely into train/val folders")
    p.add_argument("--src-images", default="hand_bbox_dataset/dataset/images", help="Source images folder")
    p.add_argument("--src-labels", default="hand_bbox_dataset/dataset/labels", help="Source labels folder")
    p.add_argument("--output-dir", default="yolo_dataset_final", help="Output folder for structured dataset")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Fraction to use for training (0-1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    p.add_argument("--move", action="store_true", help="Move files instead of copying (destructive)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output directory")
    p.add_argument("--backup", action="store_true", help="Create a timestamped backup of the source dataset before moving files")
    p.add_argument("--names-from", default="", help="Path to category_counts.json to derive class names (optional)")
    return p.parse_args()


def load_pairs(src_images: Path, src_labels: Path):
    if not src_images.exists() or not src_labels.exists():
        raise FileNotFoundError(f"Source folders not found: {src_images}, {src_labels}")

    image_extensions = (".jpg", ".jpeg", ".png")
    all_images = [p.name for p in src_images.iterdir() if p.suffix.lower() in image_extensions and p.is_file()]

    pairs = []
    missing_labels = []
    for img in all_images:
        base = Path(img).stem
        txt = f"{base}.txt"
        if (src_labels / txt).exists():
            pairs.append((img, txt))
        else:
            missing_labels.append(img)

    return pairs, missing_labels


def backup_source(src_images: Path, src_labels: Path, dest_root: Path):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = dest_root / f"backup-{ts}"
    backup_images = backup_dir / "images"
    backup_labels = backup_dir / "labels"
    backup_images.mkdir(parents=True, exist_ok=True)
    backup_labels.mkdir(parents=True, exist_ok=True)

    for p in src_images.iterdir():
        if p.is_file():
            shutil.copy2(p, backup_images / p.name)
    for p in src_labels.iterdir():
        if p.is_file():
            shutil.copy2(p, backup_labels / p.name)

    return backup_dir


def generate_names_from_counts(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Preserve original ordering from the JSON so class indices
        # match the order used when annotations were produced.
        # JSON object key insertion order is preserved by json.loads(),
        # so use that order rather than sorting alphabetically.
        names = list(data.keys())
        return names
    except Exception:
        return []


def write_data_yaml(output_dir: Path, names: list[str]):
    content = f"path: {output_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n"
    if names:
        for idx, n in enumerate(names):
            content += f"  {idx}: {n}\n"
    else:
        content += "  0: class0\n"

    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def main():
    args = parse_args()

    src_images = Path(args.src_images)
    src_labels = Path(args.src_labels)
    output_dir = Path(args.output_dir)

    pairs, missing = load_pairs(src_images, src_labels)
    total = len(pairs)

    print(f"Found {total} image/label pairs. Missing label files for {len(missing)} images.")
    if missing:
        print("Examples of images without labels:")
        for m in missing[:10]:
            print(" -", m)

    if total == 0:
        print("No valid pairs found. Aborting.")
        return 1

    random.seed(args.seed)
    random.shuffle(pairs)

    split_idx = int(total * args.train_ratio)
    train = pairs[:split_idx]
    val = pairs[split_idx:]

    print(f"Planned split -> train: {len(train)} | val: {len(val)}")

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Output directory {output_dir} already exists and is not empty.")
        if not args.overwrite:
            print("To proceed and overwrite, re-run with --overwrite")
            return 2
        else:
            # require explicit confirmation to overwrite
            resp = input("--overwrite supplied. Type OVERWRITE to proceed: ")
            if resp.strip() != "OVERWRITE":
                print("Aborted: missing OVERWRITE confirmation.")
                return 3

    if args.backup and not args.dry_run:
        print("Creating backup of source dataset (this may take time)...")
        backup_dir = backup_source(src_images, src_labels, output_dir.parent)
        print(f"Backup created at: {backup_dir}")

    # dry-run summary
    if args.dry_run:
        print("Dry-run: no files will be copied/moved.")
        return 0

    # create directories
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    op = shutil.move if args.move else shutil.copy2

    # If move is requested, require a stronger confirmation because it's destructive
    if args.move:
        resp = input("--move will REMOVE original files. Type MOVE to confirm: ")
        if resp.strip() != "MOVE":
            print("Aborted: MOVE confirmation not provided.")
            return 4

    def do_transfer(pairs_list, split_name):
        for img_name, txt_name in pairs_list:
            src_img = src_images / img_name
            src_txt = src_labels / txt_name
            dst_img = output_dir / "images" / split_name / img_name
            dst_txt = output_dir / "labels" / split_name / txt_name
            op(src_img, dst_img)
            op(src_txt, dst_txt)

    print("Copying/moving train files...")
    do_transfer(train, "train")
    print("Copying/moving val files...")
    do_transfer(val, "val")

    # generate names
    names = []
    if args.names_from:
        names = generate_names_from_counts(Path(args.names_from))
    else:
        # try default counts file next to source
        default_counts = src_images.parent.parent / "category_counts.json"
        if default_counts.exists():
            names = generate_names_from_counts(default_counts)

    write_data_yaml(output_dir, names)

    print("Split complete.")
    print(f"Train: {len(train)} | Val: {len(val)}")
    print(f"Output written to: {output_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
