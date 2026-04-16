from pathlib import Path
import threading
import time
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


def ensure_model_file() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH

    print("Downloading MediaPipe hand model...")
    urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def draw_landmarks(frame, landmarks_list, handedness_list) -> None:
    height, width = frame.shape[:2]
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    for idx, hand_landmarks in enumerate(landmarks_list):
        points = []
        for landmark in hand_landmarks:
            px = int(landmark.x * width)
            py = int(landmark.y * height)
            points.append((px, py))
            cv2.circle(frame, (px, py), 3, (0, 220, 0), -1)

        for connection in connections:
            start = points[connection.start]
            end = points[connection.end]
            cv2.line(frame, start, end, (0, 140, 255), 2)

        label = "Hand"
        if idx < len(handedness_list) and handedness_list[idx]:
            label = handedness_list[idx][0].category_name

        label_x = max(10, points[0][0] - 40)
        label_y = max(30, points[0][1] - 20)
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    model_path = ensure_model_file()

    latest_result = {
        "timestamp_ms": -1,
        "hand_landmarks": [],
        "handedness": [],
    }
    result_lock = threading.Lock()

    def on_result(result: vision.HandLandmarkerResult, _output_image: mp.Image, timestamp_ms: int) -> None:
        with result_lock:
            if timestamp_ms >= latest_result["timestamp_ms"]:
                latest_result["timestamp_ms"] = timestamp_ms
                latest_result["hand_landmarks"] = result.hand_landmarks
                latest_result["handedness"] = result.handedness

    last_sent_timestamp_ms = -1

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions and whether another app is using it.")

    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: Could not read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(now_ms, last_sent_timestamp_ms + 1)
            last_sent_timestamp_ms = timestamp_ms
            detector.detect_async(mp_image, timestamp_ms)

            with result_lock:
                hand_landmarks = latest_result["hand_landmarks"]
                handedness = latest_result["handedness"]

            if hand_landmarks:
                draw_landmarks(frame, hand_landmarks, handedness)

            cv2.putText(
                frame,
                "Hands Detector - press Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("MediaPipe Hands", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
