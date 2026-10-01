# Real-Time Driver Drowsiness Detection using YOLOv8, Haar Cascades, and CNN Classifiers

A real-time driver drowsiness detection system that combines **YOLOv8** for face detection, **Haar Cascades** for eye/mouth localization, and **lightweight CNNs** for eye-state and yawn classification. Frame-level predictions are fused over time (PERCLOS, continuous eye closure, yawn counting, MAR temporal features) into a single drowsiness score that drives a visual/audio alert. The system runs in real time on CPU-only hardware, and can optionally use a GPU for YOLO and the CNNs.

**Authors:** Lam Bao Tran (1113540) **Advisor:** Prof. Naeem Ul Islam **International Bachelor Program in Informatics (IBPI)**

## Abstract

Driver fatigue is a major factor in road accidents. This project proposes a real-time, vision-based system that uses YOLOv8 for robust face detection, Haar Cascades for region localization, and lightweight CNNs for eye-state and yawn classification. The core idea is to fuse temporal evidence (eye-closure duration, PERCLOS, yawn events, MAR-based mouth dynamics) into a stable, real-time drowsiness score, with hysteresis, debouncing and a recovery mechanism to avoid flickering alerts. An optional gaze-down guard reduces false "Closed" detections when the driver looks down.

## Model Architecture & Methodology

```
Input Frame (Webcam)
        │
        ▼
  YOLOv8 Face Detector  (every N frames, bbox smoothed with EMA)
        │
   ┌────┴─────┐
   ▼          ▼
Eye ROI     Mouth ROI
(upper 60%  (lower 2/3
 of face)    of face)
   │          │
Haar locate   Haar locate
(+ track /    (+ track /
 default ROI   default ROI
 fallback)     fallback)
   │          │
   ▼          ▼
 Eye CNN     Mouth CNN
 P(Open)     P(Yawn)
   │          │
 EMA +       MAR (bbox h/w) + temporal
 hysteresis  analysis: plateau, ZCR,
 + debounce  peaks, area, above-frac
 (+ optional (yawn vs. talking)
  gaze guard)
   │          │
   └────┬─────┘
        ▼
  Eye-closure duration · PERCLOS · Yawn count/duration
        │
        ▼
 Drowsiness score = max(eye, perclos, yawn)  ∈ [0, 1]
        │
        ▼
 Alert state machine (on/off hysteresis) + Recovery logic
        │
        ▼
 Driver Alert (red frame border + audio beep)
```

### Components

- **YOLOv8 face detector**: detects the driver's face. It runs every `detect_every_n_frames` (default 10) on a downscaled frame (default width 480), and the bounding box is smoothed with an EMA between detections.
- **Haar Cascades (eye / mouth)**: localize the eye and mouth within the face. They run on a downscaled grayscale image for speed. Because Haar cascades often miss closed eyes and wide-open mouths, the pipeline falls back to the **last known position** (`track`) or a **default ROI** (`default`), and lets the CNN decide. The eye Haar runs only every 3rd eye-pipeline run; in between, the tracked position is reused.
- **Eye CNN** (64×64): outputs `P(Open)`. The probability is smoothed with an asymmetric EMA (fast when closing, slow when reopening). A hysteresis maps it to a state: `P ≥ 0.70` → *Opened*, `P ≤ 0.40` → *Closed*, in between → keep the previous state. A *Closed* state must persist for 0.15 s (debounce) to be confirmed. When no valid eye ROI exists, the state is *Unknown* and is **not** counted as closed in PERCLOS.
- **Mouth CNN** (64×64): outputs `P(Yawn)`. "MAR" here is the **height / width ratio of the Haar mouth bounding box** (not landmark-based). When Haar misses and the fallback ROI is used, MAR is not reliable, so a pseudo-MAR derived from the CNN probability is used.
- **Temporal mouth analysis** (over a 3 s window of MAR): plateau duration, zero-crossing rate (ZCR), number of peaks, area above the MAR threshold, and fraction of time above it. These are used to separate a **yawn** (long plateau, low ZCR, large area) from **talking** (many peaks, high ZCR, short plateaus). They feed the yawn decision, not the score directly.
- **Gaze-down guard (optional, off by default)**: compares upper/lower brightness of the eye crop to detect looking down (e.g., at a keyboard) and suppresses a false *Closed* state, unless `P(Open)` is very low. Enable with `--use-gaze-guard`.

### Drowsiness score

Each component is normalized so that **1.0 = its alert threshold**; the score is their maximum:

| Component | Definition | Threshold (default) |
|---|---|---|
| Eye closure | continuous closed time / `eye_continuous_closed_sec` | 1.5 s |
| PERCLOS | fraction of closed time in a 60 s window / `perclos_thr` (only after a 20 s warm-up) | 0.30 |
| Yawn | `max(yawn_count / yawn_count_thr, current_yawn_duration / yawn_min_duration)` | 2 yawns in 60 s, or a yawn ≥ 1.0 s |

A raw alarm is raised when `score ≥ 1.0`.

### Alert state machine and recovery

- **Alert ON** after the score stays above the threshold for `on_min_sec` (1.5 s); **OFF** after it stays below for `off_min_sec` (2.0 s).
- **Recovery**: an active alert is cleared early when, for 3.0 s continuously, the eyes are *Opened*, the mouth is not yawning, PERCLOS < 0.15 (or warm-up not finished) and there was no yawn in the last 3.0 s. Old yawn events are cleared so the score does not immediately re-trigger.
- **Alert output**: a red border around the frame plus an audio beep (Windows `winsound`, macOS `afplay`, terminal bell on Linux, which may be silent depending on the terminal). The beep is muted while a recovery countdown is running, and can be disabled with `--no-audio`.

### Performance scheduling

To keep the pipeline real-time on CPU, work is spread across frames: YOLO every 10 frames, the eye CNN every frame, the mouth CNN every 2 frames (interleaved with the eye), the eye Haar every 3rd eye run, and frames are resized to width 640 by default. Cached results are reused in between.

## Testbed Setup

- Evaluated on a standard laptop with a USB webcam (640×480, 30 FPS).
- YOLOv8, Haar Cascades, and both CNNs ran on CPU.
- Dataset: ~80k eye images, ~5k mouth images, with an 80/20 train-validation split (16,190 eye and 1,023 mouth samples in validation).
- Reports eye-state and yawn training/validation accuracy, and measures end-to-end FPS under typical indoor lighting conditions.

> **Dataset Source:** the eye and mouth image datasets used for training were collected from publicly available sources online (exact origin not tracked). All images are used strictly for academic/educational purposes.

## Results

| Model     | Training Accuracy | Validation Accuracy | Validation Samples |
| --------- | ----------------- | ------------------- | ------------------ |
| Eye CNN   | ~97%              | **86%**             | 16,190             |
| Mouth CNN | ~97%              | **95%**             | 1,023              |

The Mouth CNN generalizes better than the Eye CNN. The gap between training (~97%) and validation (86%) accuracy for the Eye CNN indicates some overfitting / domain shift, which is why the pipeline adds EMA smoothing, hysteresis, debouncing and ROI fallbacks on top of the raw CNN output.

Training and validation accuracy curves (see `TrainModel Pic/`):

![Eye Model Accuracy](TrainModel%20Pic/Figure_1.png) ![Mouth Model Accuracy](TrainModel%20Pic/Figure_2.png)

*Training and validation accuracy of the eye-state CNN (left) and yawn CNN (right).*

![Validation Accuracy Comparison](TrainModel%20Pic/Figure_TrainModel.png)

*Validation accuracy comparison of the Eye and Mouth CNNs.*

**System performance:** an average of **15.7 FPS** on a CPU-only laptop. FPS depends on the scheduling options (`--detect-interval`, `--eye-every`, `--mouth-every`, `--eye-haar-every`, `--width`) and on the chosen device.

> **Limitations:** accuracy is reported per model only; there is no end-to-end evaluation of the final DROWSY/AWAKE decision yet. Haar-based MAR and the gaze-down heuristic are sensitive to lighting and camera angle.

## Conclusion

- The system combines a modern detector (YOLOv8) with efficient classical components (Haar Cascades) and lightweight CNNs, plus temporal modeling.
- It reaches real-time performance (avg. 15.7 FPS) on a CPU-only laptop.
- Eye-state and yawn CNNs reach 86% and 95% validation accuracy respectively.
- Temporal fusion (PERCLOS, eye-closure duration, yawn events), hysteresis and recovery logic make the alert more stable than frame-by-frame predictions.

## Project Structure

```
.
├── datasets/             # Training image datasets (not pushed to GitHub, see .gitignore)
│   ├── eyes/
│   │   ├── Close-Eyes/
│   │   └── Open-Eyes/
│   └── mouth/
│       ├── no yawn/
│       └── yawn/
├── Models/               # Trained models (not pushed to GitHub, see .gitignore)
│   ├── eye_detector.h5
│   ├── mouth_detector.h5
│   ├── yolov8-face.pt
│   ├── haarcascade_eye.xml
│   └── haarcascade_mcs_mouth.xml
├── TrainModel Pic/       # Training result charts (accuracy/loss)
│   ├── Figure_1.png
│   ├── Figure_2.png
│   └── Figure_TrainModel.png
├── main.py               # Runs real-time drowsiness detection via webcam
├── Train.py              # Trains the eye and mouth CNN models
├── requirements.txt
└── README.md
```

> Note: the `datasets/` and `Models/` folders are **not** included in this repo (see `.gitignore`) due to their large size. The `TrainModel Pic/` folder is kept since the images are small.

## Installation

```
git clone https://github.com/Baaotraan0407/drowsiness-detection.git
cd drowsiness-detection
pip install -r requirements.txt
```

Main dependencies: Python, OpenCV, TensorFlow/Keras, PyTorch + Ultralytics (YOLOv8), NumPy. For GPU use, install CUDA-enabled builds of PyTorch (YOLO) and/or TensorFlow (CNNs).

## Models & Dataset

The following files are required inside the `Models/` folder (not included in this repo):

- `yolov8-face.pt`: YOLOv8 face detection model
- `eye_detector.h5`: CNN for eye-state classification, 64×64 input, output = `P(Open)` (trained via `Train.py`)
- `mouth_detector.h5`: CNN for yawn classification, 64×64 input, output = `P(Yawn)` (trained via `Train.py`)
- `haarcascade_eye.xml`, `haarcascade_mcs_mouth.xml`: OpenCV Haar Cascade classifiers

The model folder is resolved in this order:

1. `--model-dir <path>` (command line)
2. the `MODEL_DIR` environment variable
3. `./Models`, then `./models` (current directory)

```
# Windows (PowerShell)
$env:MODEL_DIR="D:\path\to\Models"

# Windows (cmd)
set MODEL_DIR=D:\path\to\Models

# macOS / Linux
export MODEL_DIR=/path/to/Models
```

### Training (optional)

To retrain the eye/mouth models, prepare the dataset with this structure:

```
datasets/
├── eyes/
│   ├── Close-Eyes/
│   └── Open-Eyes/
└── mouth/
    ├── no yawn/
    └── yawn/
```

Then run:

```
python Train.py
```

The best-performing models are saved by `Train.py`; copy them into your model folder as `eye_detector.h5` and `mouth_detector.h5`.

## Usage

```
python main.py                  # asks you to choose GPU (1) or CPU (2)
python main.py --device cpu     # skip the prompt, force CPU
python main.py --device gpu     # use GPU if available (falls back to CPU with a warning)
```

- Press **q** to quit.
- The window shows: face bounding box, eye state (with `P(Open)`, EMA and ROI source), mouth state (with open duration, MAR, `P(Yawn)` and temporal features), PERCLOS, yawns in the last 60 s, status (AWAKE / DROWSY) with score, FPS and the YOLO device. A red border appears when DROWSY.
- With `--device cpu`, TensorFlow is also forced onto the CPU. With `--device gpu`, YOLO uses `cuda:0` if available, and TensorFlow uses whatever GPU its build supports.

### Command-line options

| Option | Description |
|---|---|
| `--model-dir` | Folder containing the models |
| `--camera` | Camera index (default 0) |
| `--width` | Resize frame width (default 640, 0 = keep original) |
| `--device {gpu,cpu}` | Skip the device prompt |
| `--no-audio` | Disable the audio alert |
| `--save-csv PATH` | Log time series (PERCLOS, eye/mouth state and probabilities, score, status) to CSV |
| `--log-level` | DEBUG / INFO / WARNING / ERROR |
| `--detect-interval N` | Run YOLO every N frames |
| `--yolo-downscale W` | Downscale width for YOLO (0 = off) |
| `--yolo-conf`, `--yolo-iou` | YOLO confidence / NMS IoU thresholds |
| `--eye-every N`, `--mouth-every N` | Run the eye / mouth pipeline every N frames |
| `--eye-haar-every N` | Run the eye Haar every N eye-pipeline runs |
| `--eye-open-prob-thr`, `--eye-closed-prob-thr` | Eye hysteresis thresholds (0.70 / 0.40) |
| `--eye-closed-debounce S` | Seconds a closed eye must persist (0.15) |
| `--eye-min-neighbors`, `--eye-clahe` | Eye Haar tuning / CLAHE fallback (slower) |
| `--no-eye-roi-fallback`, `--no-mouth-roi-fallback` | Disable fallback ROI when Haar misses (miss → Unknown for the eye) |
| `--use-gaze-guard` | Enable the gaze-down guard |
| `--gaze-down-ratio`, `--gaze-min-delta`, `--gaze-min-bright`, `--gaze-guard-override` | Gaze guard tuning |
| `--recovery-duration`, `--recovery-perclos`, `--recovery-yawn-cooldown` | Recovery settings (3.0 s / 0.15 / 3.0 s) |

Debug / model-compatibility options:

| Option | Description |
|---|---|
| `--show-eye-crop`, `--show-mouth-crop` | Show the crop fed to each CNN |
| `--eye-invert` | Use if the eye model outputs `P(Closed)` instead of `P(Open)` |
| `--eye-rgb`, `--mouth-rgb` | Convert BGR → RGB before the CNN (if trained on RGB images) |
| `--gray-eye`, `--gray-mouth` | Feed grayscale crops to the CNN |

> `--yolo-device` exists but is currently overridden by the GPU/CPU choice (`--device` or the prompt).

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8 / PyTorch
- TensorFlow / Keras
- NumPy, Matplotlib (training plots)