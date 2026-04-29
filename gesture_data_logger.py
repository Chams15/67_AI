from pathlib import Path
import argparse
import csv
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("hand_bbox_dataset")
DEFAULT_ANNOTATION_FILE = "annotations.csv"
COUNTER_FILE = "image_counter.txt"


def ensure_model_file() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    print("Downloading MediaPipe hand model...")
    urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture webcam frames and auto-annotate hand bounding boxes with MediaPipe. "
            "Switch category labels per frame using keyboard shortcuts."
        )
    )
    parser.add_argument(
        "--categories",
        default="gesture67,not67",
        help="Comma-separated categories (max 9). Example: open,fist,point,none",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth webcam frame (default: 1).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index passed to OpenCV VideoCapture (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested webcam frame width (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested webcam frame height (default: 720).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring of webcam frames.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where extracted frames and CSV annotations are stored.",
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=40,
        help="Padding (pixels) added around each detected hand bounding box.",
    )
    return parser.parse_args()


def parse_categories(categories_raw: str) -> list[str]:
    categories = [item.strip() for item in categories_raw.split(",") if item.strip()]
    if not categories:
        raise ValueError("At least one category is required.")
    if len(categories) > 9:
        raise ValueError("Maximum 9 categories are supported (keys 1-9).")
    return categories


def ensure_annotation_csv(csv_path: Path) -> None:
    expected_header = [
        "image_path",
        "frame_index",
        "timestamp_ms",
        "category",
        "hand_index",
        "handedness",
        "handedness_score",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            first_row = next(csv.reader(handle), None)
        if first_row == expected_header:
            return
        raise RuntimeError(
            "Annotation CSV exists with incompatible schema. "
            f"Please rename/remove: {csv_path}"
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(expected_header)


def to_hand_bbox(hand_landmarks, width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    xs = [landmark.x for landmark in hand_landmarks]
    ys = [landmark.y for landmark in hand_landmarks]

    x_min = max(0, int(min(xs) * width) - padding)
    y_min = max(0, int(min(ys) * height) - padding)
    x_max = min(width - 1, int(max(xs) * width) + padding)
    y_max = min(height - 1, int(max(ys) * height) + padding)
    return x_min, y_min, x_max, y_max


def extract_hand_boxes(result, frame_width: int, frame_height: int, padding: int) -> list[dict]:
    boxes = []
    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
        x_min, y_min, x_max, y_max = to_hand_bbox(hand_landmarks, frame_width, frame_height, padding)

        side = "unknown"
        score = 0.0
        if hand_idx < len(result.handedness) and result.handedness[hand_idx]:
            cat = result.handedness[hand_idx][0]
            side = cat.category_name.lower()
            score = float(cat.score)

        boxes.append(
            {
                "hand_index": hand_idx,
                "handedness": side,
                "handedness_score": score,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )
    return boxes


def write_yolov8_txt(txt_path: Path, boxes: list[dict], class_index: int, img_width: int, img_height: int) -> None:
    """
    Write YOLOv8-format annotations to `txt_path`.

    Each line: <class> <x_center> <y_center> <width> <height> (normalized 0..1)
    """
    lines: list[str] = []
    for box in boxes:
        x_min = box["x_min"]
        y_min = box["y_min"]
        x_max = box["x_max"]
        y_max = box["y_max"]

        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        w = (x_max - x_min) / img_width
        h = (y_max - y_min) / img_height

        lines.append(f"{class_index} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def read_image_counter(output_dir: Path) -> int:
    """Read the persistent image counter from file. Returns 1 if file doesn't exist."""
    counter_path = output_dir / COUNTER_FILE
    if counter_path.exists():
        try:
            return int(counter_path.read_text(encoding="utf-8").strip())
        except (ValueError, IOError):
            return 1
    return 1


def write_image_counter(output_dir: Path, counter: int) -> None:
    """Write the current image counter to persistent file."""
    counter_path = output_dir / COUNTER_FILE
    counter_path.write_text(str(counter), encoding="utf-8")


def draw_preview(
    frame,
    boxes: list[dict],
    current_category: str,
    categories: list[str],
    frame_idx: int,
    saved_frames: int,
    saved_boxes: int,
) -> None:
    for box in boxes:
        x_min = box["x_min"]
        y_min = box["y_min"]
        x_max = box["x_max"]
        y_max = box["y_max"]
        side = box["handedness"]

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 200, 255), 2)
        cv2.putText(
            frame,
            side,
            (x_min, max(18, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    category_hint = " | ".join([f"{idx + 1}:{name}" for idx, name in enumerate(categories)])
    cv2.putText(
        frame,
        f"Frame: {frame_idx} | Category: {current_category}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Live feed | S save frame | Q quit",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (120, 255, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Category keys: {category_hint}",
        (10, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Detected hands: {len(boxes)}",
        (10, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Saved frames: {saved_frames} | Saved boxes: {saved_boxes}",
        (10, 142),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    categories = parse_categories(args.categories)
    current_category_index = 0

    model_path = ensure_model_file()

    output_dir = Path(args.output_dir)
    dataset_dir = output_dir / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / DEFAULT_ANNOTATION_FILE
    ensure_annotation_csv(annotation_path)

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_index = -1
    saved_frames = 0
    saved_boxes = 0
    black_frame_streak = 0
    image_counter = read_image_counter(output_dir)

    with annotation_path.open("a", newline="", encoding="utf-8") as csv_handle:
        writer = csv.writer(csv_handle)

        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1
                if frame_index % args.frame_step != 0:
                    continue

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                if frame.mean() < 2.0:
                    black_frame_streak += 1
                else:
                    black_frame_streak = 0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_image)

                boxes = extract_hand_boxes(
                    result=result,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    padding=args.bbox_padding,
                )

                preview = frame.copy()
                draw_preview(
                    preview,
                    boxes=boxes,
                    current_category=categories[current_category_index],
                    categories=categories,
                    frame_idx=frame_index,
                    saved_frames=saved_frames,
                    saved_boxes=saved_boxes,
                )
                if black_frame_streak >= 5:
                    cv2.putText(
                        preview,
                        "Webcam appears black. Try --camera-index 1 or close other camera apps.",
                        (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 120, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("Video Hand BBox Annotator", preview)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"Annotation CSV: {annotation_path}")
                    print(f"Saved frames: {saved_frames}")
                    print(f"Saved hand boxes: {saved_boxes}")
                    return

                if ord("1") <= key <= ord("9"):
                    selected_index = key - ord("1")
                    if selected_index < len(categories):
                        current_category_index = selected_index

                if key == ord("s"):
                    if not boxes:
                        print(f"Skipped save for frame {frame_index}: no hands detected.")
                        continue

                    image_name = f"image{image_counter}.jpg"
                    image_path = images_dir / image_name
                    cv2.imwrite(str(image_path), frame)

                    selected_category = categories[current_category_index]
                    relative_image_path = image_path.relative_to(output_dir)
                    # Write CSV rows (legacy) and YOLOv8 .txt labels per image
                    for box in boxes:
                        writer.writerow(
                            [
                                str(relative_image_path).replace("\\", "/"),
                                frame_index,
                                frame_index,
                                selected_category,
                                box["hand_index"],
                                box["handedness"],
                                f"{box['handedness_score']:.6f}",
                                box["x_min"],
                                box["y_min"],
                                box["x_max"],
                                box["y_max"],
                            ]
                        )
                        saved_boxes += 1

                    # Create YOLOv8 annotation file alongside the saved image
                    txt_name = image_name.rsplit(".", 1)[0] + ".txt"
                    txt_path = labels_dir / txt_name
                    write_yolov8_txt(
                        txt_path=txt_path,
                        boxes=boxes,
                        class_index=current_category_index,
                        img_width=frame.shape[1],
                        img_height=frame.shape[0],
                    )

                    image_counter += 1
                    write_image_counter(output_dir, image_counter)
                    saved_frames += 1
                    print(
                        f"Saved frame {frame_index} with category '{selected_category}' "
                        f"and {len(boxes)} hand box(es). Image: {image_path}"
                    )

    cap.release()
    cv2.destroyAllWindows()

    print(f"Annotation CSV: {annotation_path}")
    print(f"Saved frames: {saved_frames}")
    print(f"Saved hand boxes: {saved_boxes}")


if __name__ == "__main__":
    main()
