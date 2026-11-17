"""Patient-care ASL communicator.

The classifier recognises single letters. This script watches for letters that
are held steadily, remembers the last few, and when they spell one of the
COMBOS below it takes over the screen with an alert.

Keys:  b = back
       q = quit   f = fullscreen
created by Nikita Li-Tsun-Ye
"""
import os
import json
import time

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from asl_common import (BASE_DIR, create_landmarker, landmarks_to_features,
                        draw_skeleton, open_camera)

MODEL_FILE = os.path.join(BASE_DIR, "asl_landmark_model.keras")
CLASSES_FILE = os.path.join(BASE_DIR, "asl_classes.json")
CAMERA_INDEX = 0
WINDOW = "ASL Patient Communicator"

MIN_CONFIDENCE = 0.70   
HOLD_SECONDS = 0.7       
SEQUENCE_TIMEOUT = 5.0   
FLASH_PERIOD = 0.3       

RED, GREEN, WHITE = (0, 0, 255), (0, 200, 0), (255, 255, 255)

COMBOS = {
    "sos": {"text": "EMERGENCY",     "style": "flash", "colors": [RED, GREEN],
            "text_color": WHITE, "duration": None},
    "wa":  {"text": "WATER\nPLEASE", "style": "solid", "colors": [(200, 110, 0)],
            "text_color": WHITE, "duration": 10},
    "co":  {"text": "TOO\nCOLD",     "style": "solid", "colors": [(255, 225, 140)],
            "text_color": (110, 40, 0), "duration": 10},
    "ho":  {"text": "TOO\nHOT",      "style": "solid", "colors": [(0, 90, 255)],
            "text_color": WHITE, "duration": 10},
    "rr":  {"text": "RESTROOM pls",            "style": "solid", "colors": [(40, 40, 40)],
            "text_color": (0, 255, 255), "duration": 10},
    "nu": {"text": "NURSE\nNEEDED", "style": "solid", "colors": [(0, 160, 0)],
       "text_color": (255, 255, 255), "duration": 10},
    "hu": {"text": "HUNGRY", "style": "solid", "colors": [(200, 160, 0)],
       "text_color": (0, 255, 255), "duration": 10},
}

FONT = cv2.FONT_HERSHEY_DUPLEX

class LetterCommitter:
    
    def __init__(self):
        self.candidate, self.since, self.locked = None, 0.0, False

    def update(self, letter, conf, now):
        """Returns (committed_letter_or_None, hold_progress_0_to_1)."""
        if letter is None or conf < MIN_CONFIDENCE:
            self.candidate, self.locked = None, False
            return None, 0.0
        if letter != self.candidate:
            self.candidate, self.since, self.locked = letter, now, False
        if self.locked:
            return None, 1.0
        progress = min((now - self.since) / HOLD_SECONDS, 1.0)
        if progress >= 1.0:
            self.locked = True
            return letter, 1.0
        return None, progress


class SequenceMatcher:
    def __init__(self, patterns):
        self.patterns = sorted(patterns, key=len, reverse=True)  
        self.max_len = max(len(p) for p in patterns)
        self.items = []                                           

    def _prune(self, now):
        self.items = [(l, t) for l, t in self.items if now - t <= SEQUENCE_TIMEOUT]

    def push(self, letter, now):
        self._prune(now)
        self.items = (self.items + [(letter, now)])[-self.max_len:]
        typed = "".join(l for l, _ in self.items)
        for p in self.patterns:
            if typed.endswith(p):
                self.items.clear()
                return p
        return None

    def typed(self, now):
        self._prune(now)
        return "".join(l for l, _ in self.items)


def draw_big_text(frame, text, color):
    h, w = frame.shape[:2]
    lines = text.split("\n")
    widest = max(cv2.getTextSize(l, FONT, 1.0, 2)[0][0] for l in lines)
    line_h1 = cv2.getTextSize("H", FONT, 1.0, 2)[0][1]
    scale = min(0.9 * w / widest, 0.8 * h / (len(lines) * line_h1 * 1.5))
    thick = max(2, int(scale * 2.2))
    line_h = cv2.getTextSize("H", FONT, scale, thick)[0][1]
    gap = int(line_h * 0.45)
    y = (h - (len(lines) * line_h + (len(lines) - 1) * gap)) // 2 + line_h
    for l in lines:
        x = (w - cv2.getTextSize(l, FONT, scale, thick)[0][0]) // 2
        cv2.putText(frame, l, (x, y), FONT, scale, (0, 0, 0), thick + 8, cv2.LINE_AA)
        cv2.putText(frame, l, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
        y += line_h + gap


def draw_back_button(frame):
    w = frame.shape[1]
    label = "< BACK (B)"
    scale = max(0.7, w / 640 * 0.9)
    thick = max(2, int(scale * 2))
    (tw, th), base = cv2.getTextSize(label, FONT, scale, thick)
    pad = int(14 * scale)
    x1, y1 = 20, 20
    x2, y2 = x1 + tw + 2 * pad, y1 + th + base + 2 * pad
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv2.putText(frame, label, (x1 + pad, y1 + pad + th), FONT, scale,
                (255, 255, 255), thick, cv2.LINE_AA)
    return (x1, y1, x2, y2)


def render_alert(frame, combo, elapsed):
    colors = combo["colors"]
    if combo["style"] == "flash":
        frame[:] = colors[int(elapsed / FLASH_PERIOD) % len(colors)]
    else:
        frame[:] = colors[0]
    draw_big_text(frame, combo["text"], combo["text_color"])
    hint = "Press B or click BACK to return to the camera"
    pos = (20, frame.shape[0] - 20)
    cv2.putText(frame, hint, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, hint, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return draw_back_button(frame)


def draw_hud(frame, letter, conf, progress, typed):
    h, w = frame.shape[:2]
    if letter:
        col = (0, 220, 0) if conf >= MIN_CONFIDENCE else (0, 165, 255)
        cv2.putText(frame, f"{letter.upper()}  {conf * 100:.0f}%", (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Show your hand...", (25, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (25, 70), (225, 86), (255, 255, 255), 1)             # hold bar
    cv2.rectangle(frame, (25, 70), (25 + int(200 * progress), 86), (0, 220, 0), -1)
    cv2.putText(frame, "Signed: " + " ".join(typed.upper()), (25, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y = h - 14 - 24 * (len(COMBOS) - 1)                                       # legend
    for pattern, combo in COMBOS.items():
        cv2.putText(frame, f"{'-'.join(pattern.upper())} = {combo['text'].replace(chr(10), ' ')}",
                    (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(CLASSES_FILE)):
        print("Model not found. Run  python train_landmarks.py  first.")
        return
    with open(CLASSES_FILE) as f:
        classes = [c.lower() for c in json.load(f)]
    missing = sorted(set("".join(COMBOS)) - set(classes))
    if missing:
        print(f"WARNING: the model doesn't know the letters {missing}; "
              f"signs using them will never trigger.", flush=True)

    model = tf.keras.models.load_model(MODEL_FILE)
    landmarker = create_landmarker(video=True)
    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        print("ERROR: could not open a camera (close Zoom/Teams/browser tabs, check permissions).")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    ui = {"back_rect": None, "back_clicked": False}   

    def on_mouse(event, mx, my, flags, param):
        r = ui["back_rect"]
        if event == cv2.EVENT_LBUTTONUP and r and r[0] <= mx <= r[2] and r[1] <= my <= r[3]:
            ui["back_clicked"] = True

    cv2.setMouseCallback(WINDOW, on_mouse)
    fullscreen = False
    committer, matcher = LetterCommitter(), SequenceMatcher(COMBOS.keys())
    history = []
    alert = None                                  # {"combo": ..., "start": ...}
    start, last_ts = time.monotonic(), -1
    print("Running. q = quit, x = dismiss alert, f = fullscreen.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.monotonic()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((now - start) * 1000), last_ts + 1)
        last_ts = ts
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        letter, conf = None, 0.0
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            history.append(model(landmarks_to_features(lms, w, h)[None], training=False).numpy()[0])
            history = history[-7:]              
            avg = np.mean(history, axis=0)
            best = int(np.argmax(avg))
            letter, conf = classes[best], float(avg[best])
            draw_skeleton(frame, lms)
        else:
            history.clear()

        committed, progress = committer.update(letter, conf, now)
        if committed:
            pattern = matcher.push(committed, now)
            if pattern:
                combo = COMBOS[pattern]
                sticky_active = alert is not None and alert["combo"]["duration"] is None
                if not sticky_active or combo["duration"] is None:   # EMERGENCY can't be overridden
                    alert = {"combo": combo, "start": now}
                    print(f"ALERT: {combo['text'].replace(chr(10), ' ')}", flush=True)

        if alert and alert["combo"]["duration"] is not None \
                and now - alert["start"] > alert["combo"]["duration"]:
            alert = None

        if ui["back_clicked"]:                     
            ui["back_clicked"] = False
            if alert:
                alert = None
                matcher.items.clear()
                print("Back to camera.", flush=True)

        if alert:
            ui["back_rect"] = render_alert(frame, alert["combo"], now - alert["start"])
        else:
            ui["back_rect"] = None                    
            draw_hud(frame, letter, conf, progress, matcher.typed(now))

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("b"), ord("B"), ord("x"), ord("X"), 8, 27): 
            if alert:
                alert = None
                matcher.items.clear()              
                print("Back to camera.", flush=True)
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()