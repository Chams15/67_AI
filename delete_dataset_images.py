from pathlib import Path
import argparse
import csv
import json
import sys


DEFAULT_OUTPUT_DIR = Path(__file__).with_name("hand_bbox_dataset")
COUNTS_FILE = "category_counts.json"
ANNOTATION_FILE = "annotations.csv"


def parse_id_list(id_str: str) -> set[int]:
    """Parse strings like '1,3,5-7' into a set of ints."""
    ids = set()
    if not id_str:
        return ids
    for part in id_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a_i = int(a)
                b_i = int(b)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid range: {part}")
            if b_i < a_i:
                raise argparse.ArgumentTypeError(f"Invalid range: {part}")
            ids.update(range(a_i, b_i + 1))
        else:
            try:
                ids.add(int(part))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid id: {part}")
    return ids


def load_category_counts(output_dir: Path) -> dict:
    p = output_dir / COUNTS_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_category_counts(output_dir: Path, counts: dict) -> None:
    p = output_dir / COUNTS_FILE
    p.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete images and their annotations from dataset")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder used by the logger")
    parser.add_argument("--ids", default="", help="Comma-separated image ids or ranges to delete, e.g. 1,3,5-7")
    parser.add_argument("--filenames", default="", help="Comma-separated exact image filenames to delete (e.g. image12.jpg)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "dataset" / "images"
    labels_dir = output_dir / "dataset" / "labels"
    annotation_path = output_dir / ANNOTATION_FILE

    ids = parse_id_list(args.ids)
    filenames = [s.strip() for s in args.filenames.split(",") if s.strip()]

    targets = set()
    for i in ids:
        targets.add(f"image{i}.jpg")
    for fn in filenames:
        targets.add(fn)

    if not targets:
        print("No targets specified. Use --ids or --filenames.")
        return 1

    print("Targets to delete:")
    for t in sorted(targets):
        print(" -", t)
    if not args.yes:
        resp = input("Proceed and delete these files and annotations? [y/N]: ")
        if resp.lower() != "y":
            print("Aborted")
            return 0

    # Read CSV rows
    kept_rows = []
    removed_images = set()
    removed_image_to_category = {}
    if annotation_path.exists():
        with annotation_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header:
                kept_rows.append(header)
            for row in reader:
                if not row:
                    continue
                image_path = row[0].replace("\\", "/")
                image_name = Path(image_path).name
                if image_name in targets:
                    removed_images.add(image_name)
                    # capture category for this image if not yet recorded
                    category = row[3] if len(row) > 3 else None
                    if image_name not in removed_image_to_category and category:
                        removed_image_to_category[image_name] = category
                    continue
                kept_rows.append(row)

    # Delete files
    deleted = []
    for image_name in targets:
        img_path = images_dir / image_name
        lbl_name = Path(image_name).with_suffix('.txt').name
        lbl_path = labels_dir / lbl_name
        did = False
        if img_path.exists():
            try:
                img_path.unlink()
                did = True
            except Exception as e:
                print(f"Failed to delete image {img_path}: {e}")
        if lbl_path.exists():
            try:
                lbl_path.unlink()
                did = True
            except Exception as e:
                print(f"Failed to delete label {lbl_path}: {e}")
        if did:
            deleted.append(image_name)

    # Write updated CSV
    if annotation_path.exists():
        tmp = annotation_path.with_suffix('.tmp')
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for row in kept_rows:
                writer.writerow(row)
        tmp.replace(annotation_path)

    # Update category counts
    counts = load_category_counts(output_dir)
    for image_name, cat in removed_image_to_category.items():
        if cat in counts:
            counts[cat] = max(0, counts[cat] - 1)
    save_category_counts(output_dir, counts)

    # Report
    print(f"Deleted {len(deleted)} files:")
    for d in deleted:
        print(" -", d)
    missing = targets - set(deleted)
    if missing:
        print("Not found (or not deleted):")
        for m in sorted(missing):
            print(" -", m)

    print("Updated annotations and category counts.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
