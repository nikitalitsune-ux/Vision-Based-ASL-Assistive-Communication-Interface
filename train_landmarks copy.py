import os
import json
from collections import Counter

import cv2
import numpy as np
import kagglehub
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks

from asl_common import (BASE_DIR, create_landmarker, landmarks_to_features,
                        normalize_points)

INCLUDE_DIGITS = False   
AUG_PER_SAMPLE = 30      
MIN_SAMPLES = 10        
SEED = 42

PRIORITY_LETTERS = {"s", "o", "w", "a", "c", "h"}
PRIORITY_WEIGHT = 3.0

MODEL_OUT = os.path.join(BASE_DIR, "asl_landmark_model.keras")
CLASSES_OUT = os.path.join(BASE_DIR, "asl_classes.json")

def find_images(root):
    found = {}
    for folder, _, files in os.walk(root):
        cls = os.path.basename(folder)
        if cls.isdigit() and not INCLUDE_DIGITS:
            continue
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                found.setdefault((cls, f), os.path.join(folder, f))
    return [(cls, path) for (cls, _), path in sorted(found.items())]


def extract_features(landmarker, img):
    pad = int(0.3 * max(img.shape[:2]))
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for attempt in (img, padded, cv2.flip(padded, 1)):
        rgb = cv2.cvtColor(attempt, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.hand_landmarks:
            h, w = attempt.shape[:2]
            return landmarks_to_features(result.hand_landmarks[0], w, h)
    return None


def build_feature_table(root):
    landmarker = create_landmarker(video=False, detection_conf=0.3)
    items = find_images(root)
    print(f"Found {len(items)} images. Detecting hands (this takes a minute or two)...", flush=True)

    feats, labels, missed = [], [], Counter()
    for i, (cls, path) in enumerate(items, 1):
        img = cv2.imread(path)
        feat = extract_features(landmarker, img) if img is not None else None
        if feat is None:
            missed[cls] += 1
        else:
            feats.append(feat)
            labels.append(cls)
        if i % 300 == 0:
            print(f"  {i}/{len(items)}", flush=True)
    landmarker.close()

    print(f"Hand found in {len(feats)}/{len(items)} images.", flush=True)
    if missed:
        print("Images with no hand detected, per class:", dict(missed), flush=True)
    return np.array(feats, dtype=np.float32), np.array(labels)

def rotation_matrix(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def mirror(feat):
    pts = feat.reshape(21, 3).copy()
    pts[:, 0] *= -1                   
    return pts.flatten()


def augment(feat, rng):
    pts = feat.reshape(21, 3).astype(np.float64)
    if rng.random() < 0.5:
        pts[:, 0] *= -1
    angles = np.deg2rad(rng.uniform([-10, -10, -15], [10, 10, 15]))   # small tilts only
    pts = pts @ rotation_matrix(*angles).T
    pts *= rng.uniform(0.9, 1.1, size=3)
    pts += rng.normal(0, 0.01, pts.shape)
    return normalize_points(pts).flatten()


def expand_training_set(X, y, rng):
    Xs, ys = [X, np.array([mirror(f) for f in X])], [y, y]
    for _ in range(AUG_PER_SAMPLE):
        Xs.append(np.array([augment(f, rng) for f in X]))
        ys.append(y)
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def main():
    rng = np.random.default_rng(SEED)
    tf.random.set_seed(SEED)

    print("Locating ASL dataset...", flush=True)
    root = kagglehub.dataset_download("ayuraj/asl-dataset")
    X, labels = build_feature_table(root)

    counts = Counter(labels)
    classes = sorted(c for c, n in counts.items() if n >= MIN_SAMPLES)
    dropped = sorted(set(counts) - set(classes))
    if dropped:
        print(f"WARNING: dropping classes with too few detections: {dropped}", flush=True)
    keep = np.isin(labels, classes)
    X, labels = X[keep], labels[keep]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in labels])
    print(f"Training on {len(classes)} classes: {classes}", flush=True)

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.15, random_state=SEED, stratify=y_tv)
    X_train, y_train = expand_training_set(X_train, y_train, rng)
    print(f"Train {len(X_train)} (augmented), val {len(X_val)}, test {len(X_test)}", flush=True)

    model = models.Sequential([
        layers.Input(shape=(63,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(classes), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    class_weight = {i: (PRIORITY_WEIGHT if c in PRIORITY_LETTERS else 1.0)
                    for c, i in class_to_idx.items()}
    missing_priority = sorted(PRIORITY_LETTERS - set(classes))
    if missing_priority:
        print(f"WARNING: priority letters missing from the data: {missing_priority}", flush=True)

    model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=64, verbose=2,
        class_weight=class_weight,
        callbacks=[callbacks.EarlyStopping(patience=12, restore_best_weights=True),
                   callbacks.ReduceLROnPlateau(factor=0.5, patience=5)],
    )

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc * 100:.1f}%", flush=True)
    pred = model.predict(X_test, verbose=0).argmax(axis=1)

    print("Accuracy on the patient-care letters (held-out images):", flush=True)
    for c in sorted(PRIORITY_LETTERS & set(classes)):
        mask = y_test == class_to_idx[c]
        if mask.sum():
            print(f"  {c.upper()}: {(pred[mask] == class_to_idx[c]).mean() * 100:.0f}%"
                  f"  ({mask.sum()} test images)", flush=True)
    confusions = Counter((classes[t], classes[p]) for t, p in zip(y_test, pred) if t != p)
    if confusions:
        print("Most common mix-ups (true -> predicted):", flush=True)
        for (t, p), n in confusions.most_common(8):
            print(f"  {t.upper()} -> {p.upper()}  x{n}", flush=True)

    model.save(MODEL_OUT)
    with open(CLASSES_OUT, "w") as f:
        json.dump(classes, f)
    print(f"\nSaved {MODEL_OUT}\nSaved {CLASSES_OUT}", flush=True)


if __name__ == "__main__":
    main()
