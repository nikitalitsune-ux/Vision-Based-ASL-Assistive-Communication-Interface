
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_FILE = os.path.join(BASE_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
]


def ensure_hand_model():
    if not os.path.exists(HAND_MODEL_FILE):
        print("Downloading hand tracking model (one time, ~8 MB)...", flush=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_FILE)
    return HAND_MODEL_FILE


def create_landmarker(video=False, detection_conf=0.5):
    """video=False -> for still images (training); video=True -> for the webcam."""
    with open(ensure_hand_model(), "rb") as f:
        model_bytes = f.read()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=detection_conf,
        min_hand_presence_confidence=detection_conf,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def normalize_points(pts):
    """pts: (21, 3) array. Moves the wrist to the origin and rescales so the
    result doesn't depend on where the hand is or how far it is from the camera."""
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    return (pts / max(scale, 1e-6)).astype(np.float32)


def landmarks_to_features(landmarks, w, h):
    """21 MediaPipe landmarks -> 63 numbers (x, y, z per joint)."""
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
    return normalize_points(pts).flatten()


def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 4, cv2.LINE_AA)   # green pipes
    for p in pts:
        cv2.circle(frame, p, 6, (0, 0, 255), -1, cv2.LINE_AA)          # red joints


def open_camera(preferred=0):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for index in dict.fromkeys([preferred, 0, 1, 2]):
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"Camera opened (index {index}).", flush=True)
                    return cap
            cap.release()
    return None