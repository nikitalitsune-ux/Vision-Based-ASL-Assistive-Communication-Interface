import os
import json
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from asl_common import (BASE_DIR, create_landmarker, landmarks_to_features,
                        draw_skeleton, open_camera)

MODEL_FILE = os.path.join(BASE_DIR, "asl_landmark_model.keras")
CLASSES_FILE = os.path.join(BASE_DIR, "asl_classes.json")
CAMERA_INDEX = 0
MIN_CONFIDENCE = 0.50    


def main():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(CLASSES_FILE)):
        print("Model not found. Run  python train_landmarks.py  first.")
        return

    with open(CLASSES_FILE) as f:
        classes = json.load(f)
    model = tf.keras.models.load_model(MODEL_FILE)
    landmarker = create_landmarker(video=True)

    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        print("ERROR: could not open a camera (close Zoom/Teams/browser tabs, check permissions).")
        return

    print("Running. Press 'q' in the video window to quit.", flush=True)
    history = deque(maxlen=7)
    start, last_ts = time.monotonic(), -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            feat = landmarks_to_features(lms, w, h)
            history.append(model(feat[None], training=False).numpy()[0])
            avg = np.mean(history, axis=0)
            top = np.argsort(avg)[::-1][:3]

            draw_skeleton(frame, lms)

            best, conf = classes[top[0]].upper(), avg[top[0]]
            if conf >= MIN_CONFIDENCE:
                cv2.putText(frame, f"{best}  {conf * 100:.0f}%", (30, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "?", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (0, 165, 255), 3, cv2.LINE_AA)
            for row, i in enumerate(top):
                cv2.putText(frame, f"{classes[i].upper()}: {avg[i] * 100:.0f}%",
                            (30, 95 + row * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
        else:
            history.clear()
            cv2.putText(frame, "Show your hand...", (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("ASL Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()