
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # save figures instead of opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = "datasets"
MODEL_DIR = "models"
RESULTS_DIR = "results"
IMG_SIZE = (64, 64)
BATCH = 32
SEED = 42
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
SUBJECT_RE = re.compile(r"^(s\d+)_", re.IGNORECASE)  # e.g. s0001_00001_...

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def list_images(root, classes):
    """Return a DataFrame with one row per image: filepath, label, subject."""
    rows = []
    for label in classes:
        folder = os.path.join(root, label)
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(IMG_EXT):
                m = SUBJECT_RE.match(name)
                rows.append({"filepath": os.path.join(folder, name),
                             "label": label,
                             "subject": m.group(1).lower() if m else None})
    return pd.DataFrame(rows)


def split(df, name):
    """Split by subject when every file has a subject ID, otherwise randomly."""
    if df["subject"].notna().all():
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, va = next(gss.split(df, groups=df["subject"]))
        train, val = df.iloc[tr], df.iloc[va]
        print(f"[{name}] subject-independent split: "
              f"{train.subject.nunique()} subjects train / {val.subject.nunique()} val")
        mode = "subject-independent"
    else:
        train, val = train_test_split(df, test_size=0.2, stratify=df["label"],
                                      random_state=SEED)
        print(f"[{name}] WARNING: no subject IDs in file names -> random split. "
              "The same person may appear in train and validation.")
        mode = "random (no subject IDs)"
    return train, val, mode


def make_generators(train_df, val_df, classes, augment):
    train_gen = ImageDataGenerator(rescale=1. / 255, **augment)
    val_gen = ImageDataGenerator(rescale=1. / 255)  # no augmentation
    common = dict(x_col="filepath", y_col="label", target_size=IMG_SIZE,
                  class_mode="binary", classes=classes, batch_size=BATCH)
    train = train_gen.flow_from_dataframe(train_df, shuffle=True, seed=SEED, **common)
    val = val_gen.flow_from_dataframe(val_df, shuffle=False, **common)
    return train, val


def build_eye_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2, 2), Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2), Dropout(0.25),
        Flatten(), Dense(128, activation="relu"), Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_mouth_model():
    model = Sequential([
        Conv2D(32, (5, 5), activation="relu", input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (5, 5), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(), Dense(128, activation="relu"), Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate(model, val, classes, positive):
    """Metrics on the un-augmented validation set. `positive` is the class
    whose recall matters most (the one we must not miss)."""
    val.reset()
    prob = model.predict(val, verbose=0).ravel()          # P(classes[1])
    y_true = val.classes                                   # 0 / 1 in classes order
    y_pred = (prob >= 0.5).astype(int)
    pos = classes.index(positive)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n_val": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        f"precision_{positive}": round(float(precision_score(y_true, y_pred, pos_label=pos)), 4),
        f"recall_{positive}": round(float(recall_score(y_true, y_pred, pos_label=pos)), 4),
        f"f1_{positive}": round(float(f1_score(y_true, y_pred, pos_label=pos)), 4),
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": cm.tolist()},
    }


def train_one(name, root, classes, positive, builder, augment, out_file):
    df = list_images(root, classes)
    print(f"[{name}] {len(df)} images: {df.label.value_counts().to_dict()}")
    train_df, val_df, mode = split(df, name)
    train, val = make_generators(train_df, val_df, classes, augment)

    model = builder()
    ckpt = os.path.join(MODEL_DIR, out_file)
    history = model.fit(
        train, validation_data=val, epochs=50,
        callbacks=[EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
                   ModelCheckpoint(ckpt, save_best_only=True)])

    metrics = evaluate(model, val, classes, positive)
    metrics["split"] = mode
    metrics["n_train"] = int(len(train_df))
    print(f"[{name}] {json.dumps(metrics, indent=2)}")
    return history, metrics


def plot(histories):
    plt.figure(figsize=(12, 5))
    for i, (title, h) in enumerate(histories.items(), 1):
        plt.subplot(1, 2, i)
        plt.plot(h.history["accuracy"], label="Train")
        plt.plot(h.history["val_accuracy"], label="Validation (no augmentation)")
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150)


def main():
    eye_hist, eye_m = train_one(
        "eye", os.path.join(DATASET_DIR, "eyes"), ["Close-Eyes", "Open-Eyes"],
        positive="Close-Eyes",   # missing a closed eye is the costly error
        builder=build_eye_model, out_file="eye_detector.h5",
        augment=dict(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                     shear_range=0.1, zoom_range=0.1, horizontal_flip=False))

    mouth_hist, mouth_m = train_one(
        "mouth", os.path.join(DATASET_DIR, "mouth"), ["no yawn", "yawn"],
        positive="yawn",
        builder=build_mouth_model, out_file="mouth_detector.h5",
        augment=dict(rotation_range=10, width_shift_range=0.15, height_shift_range=0.1,
                     shear_range=0.1, zoom_range=0.2, horizontal_flip=True))

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"eye": eye_m, "mouth": mouth_m}, f, indent=2)
    plot({"Eye model": eye_hist, "Mouth model": mouth_hist})
    print(f"Saved {RESULTS_DIR}/metrics.json and {RESULTS_DIR}/training_curves.png")


if __name__ == "__main__":
    main()