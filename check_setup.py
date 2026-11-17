"""Health check for the ASL Patient Communicator.

Run it after installing, or whenever something doesn't work:

    python check_setup.py               # full check (briefly opens the camera)
    python check_setup.py --no-camera   # skip the camera test
"""
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
problems = 0


def ok(msg):
    print(f"  [ OK ]  {msg}")


def warn(msg):
    print(f"  [WARN]  {msg}")


def fail(msg, fix):
    global problems
    problems += 1
    print(f"  [FAIL]  {msg}\n          -> {fix}")


def check_python():
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro}"
    if (3, 10) <= (v.major, v.minor) <= (3, 12):
        ok(text)
    else:
        warn(f"{text}: Python 3.10-3.12 is recommended. Other versions may have no "
             f"TensorFlow/MediaPipe packages yet, which makes pip install fail.")


def check_imports():
    loaded = {}
    required = [("numpy", "numpy"), ("cv2", "OpenCV (installed together with mediapipe)"),
                ("mediapipe", "mediapipe"), ("tensorflow", "tensorflow")]
    for module, label in required:
        try:
            loaded[module] = __import__(module)
            ok(f"{label} {getattr(loaded[module], '__version__', '')}".strip())
        except Exception as e:
            fail(f"Could not import {label}: {type(e).__name__}: {e}",
                 "Activate your virtual environment, then: pip install -r requirements.txt")
    for module in ("sklearn", "kagglehub"):
        try:
            __import__(module)
            ok(f"{module} (needed only for training)")
        except Exception:
            warn(f"{module} not installed - only needed if you want to retrain the model")
    return loaded


def check_hand_tracker(loaded):
    if "mediapipe" not in loaded:
        return
    try:
        from mediapipe.tasks.python import vision
        vision.HandLandmarker  # noqa: B018 - just checking it exists
        ok("MediaPipe HandLandmarker is available")
    except Exception:
        fail("This MediaPipe version has no HandLandmarker", "pip install -U mediapipe")
        return
    try:
        from asl_common import ensure_hand_model
        path = ensure_hand_model()
        ok(f"Hand tracking model ready ({os.path.basename(path)})")
    except Exception as e:
        fail(f"Could not get the hand tracking model: {e}",
             "Check your internet connection, or download hand_landmarker.task manually "
             "(the URL is at the top of asl_common.py) into this folder")


def check_model(loaded):
    model_file = os.path.join(BASE_DIR, "asl_landmark_model.keras")
    classes_file = os.path.join(BASE_DIR, "asl_classes.json")
    if not (os.path.exists(model_file) and os.path.exists(classes_file)):
        fail("asl_landmark_model.keras and/or asl_classes.json are missing",
             "Run: python train_landmarks.py   (needs internet; takes a few minutes)")
        return
    with open(classes_file) as f:
        classes = [c.lower() for c in json.load(f)]
    ok(f"asl_classes.json lists {len(classes)} letters")
    if "tensorflow" not in loaded:
        return
    try:
        model = loaded["tensorflow"].keras.models.load_model(model_file)
    except Exception as e:
        fail(f"The model file would not load: {type(e).__name__}: {e}",
             "Your TensorFlow is probably older than the one that saved the model. "
             "pip install -U tensorflow   -or-   retrain: python train_landmarks.py")
        return
    if model.input_shape[-1] != 63 or model.output_shape[-1] != len(classes):
        fail(f"Model shape {model.input_shape} -> {model.output_shape} doesn't match "
             f"{len(classes)} letters", "Retrain: python train_landmarks.py")
        return
    ok("Model loads and matches the letter list")
    try:
        from predict_patient import COMBOS
        needed = sorted(set("".join(COMBOS)) - set(classes))
        if needed:
            fail(f"Signs in COMBOS use letters the model doesn't know: {needed}",
                 "Add them to PRIORITY_LETTERS in train_landmarks.py and retrain")
        else:
            ok(f"All letters used by the {len(COMBOS)} signs are known to the model")
    except Exception as e:
        warn(f"Could not check the signs: {e}")


def check_camera(loaded):
    if "--no-camera" in sys.argv:
        warn("Camera test skipped")
        return
    if "cv2" not in loaded:
        return
    try:
        from asl_common import open_camera
        cap = open_camera(0)
        if cap is None:
            fail("No camera could be opened",
                 "Close Zoom/Teams/browser tabs using it, allow camera access for your "
                 "terminal/IDE (macOS: System Settings > Privacy > Camera), or change CAMERA_INDEX")
        else:
            cap.release()
            ok("Camera works")
    except Exception as e:
        fail(f"Camera test crashed: {e}", "See the troubleshooting section in README.md")


def main():
    print("\nASL Patient Communicator - setup check\n")
    check_python()
    loaded = check_imports()
    check_hand_tracker(loaded)
    check_model(loaded)
    check_camera(loaded)
    print()
    if problems:
        print(f"{problems} problem(s) found. Fix the [FAIL] lines above and run this again.")
    else:
        print("Everything looks good. Start it with:  python predict_patient.py")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
