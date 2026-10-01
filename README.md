# Real-Time Driver Drowsiness Detection using YOLOv8, Haar Cascades, and CNN Classifiers

A real-time driver drowsiness detection system that combines **YOLOv8** for face detection, **Haar Cascades** for eye/mouth region extraction, and **lightweight CNNs** for eye-state and yawn classification. The system fuses temporal features (MAR, PERCLOS) with eye-closure duration and yawn counting to produce a stable drowsiness score. It runs in real time on a CPU-only laptop, and YOLOv8 can optionally run on an NVIDIA GPU (selectable at startup).

**Authors:** Lam Bao Tran (1113540) **Advisor:** Prof. Naeem Ul Islam **International Bachelor Program in Informatics (IBPI)**

## Abstract

Driver fatigue is a major factor in road accidents. This project proposes a real-time, vision-based system that uses YOLOv8 for robust face detection, Haar Cascades for region extraction, and lightweight CNNs for eye-state and yawn classification. The core idea is fusing temporal features (MAR, PERCLOS), eye-closure duration and yawn counting into a single drowsiness score, with a recovery mechanism that clears the alert once the driver is attentive again.

## Model Architecture & Methodology

The pipeline is a sequence of modular components that runs in real time:

```
Input Frame (Webcam)
        │
        ▼
  YOLOv8 Face Detector   (CPU or GPU, chosen at startup)
        │
   ┌────┴────┐
   ▼         ▼
Haar Cascade   Haar Cascade
 Eye ROI        Mouth ROI
(+ ROI tracking / default ROI fallback when Haar misses)
   │              │
   ▼              ▼
 Eye CNN       Mouth CNN
 P(Open)       P(Yawn)
   │              │
   │        Temporal Features (MAR, ZCR, Peaks,
   │        Plateau, Above-threshold Area)
   │              │
   └──────┬───────┘
          ▼
 PERCLOS + Eye-closure time + Yawn count
   (Drowsiness Score = max of the three)
          │
          ▼
  Alert state machine (on/off hysteresis + recovery)
          │
          ▼
   Driver Alert (Visual border / Audio beep)
```

- **YOLOv8 Face Detector**: detects the driver's face. It runs every N frames (default 10) on a downscaled frame, and the bounding box is smoothed with an EMA in between.
- **Haar Cascades**: locate the eye ROI and mouth ROI inside the face. Because Haar cascades are trained on open eyes/closed mouths, they often miss when the eyes are closed or the mouth is wide open. In that case the system reuses the last known position (`track`) or a default ROI (`default`) and lets the CNN decide.
- **Two lightweight CNNs (64×64 input)**: classify eye state (output = P(Open)) and yawn (output = P(Yawn)).
- **Eye logic**: EMA smoothing (fast when the eye closes, slow when it reopens), hysteresis thresholds (Opened ≥ 0.70, Closed ≤ 0.40) and a short debounce before `Closed` is confirmed. An optional **gaze-down guard** (off by default) suppresses false `Closed` readings when the driver looks down at a keyboard or phone.
- **Mouth logic**: mouth aspect ratio (MAR) over a 3-second window gives plateau duration, zero-crossing rate, peak count and above-threshold area, which separate **yawning** (long plateau, low ZCR) from **talking** (many peaks, high ZCR).
- **Drowsiness score** in [0, 1]: `score = max(eye, PERCLOS, yawn)`, where each component is normalized so that 1.0 is exactly its alert threshold:
  - **eye**: continuous eye-closure time / 1.5 s
  - **PERCLOS**: percentage of closed-eye time over a 60 s window / 0.30 (only after a 20 s warm-up)
  - **yawn**: yawns in the last 60 s / 2, or the duration of a yawn in progress / 1 s
- **Alert state machine**: the alert turns on after the score stays ≥ 1.0 for 1.5 s and turns off after it stays below for 2.0 s.
- **Recovery**: the alert is cleared early when, for 3 s in a row, the eyes are open, the mouth is not yawning, PERCLOS < 0.15 and there has been no yawn for 3 s. Old yawn events are then cleared so the alert does not immediately re-trigger. The beep is muted while recovery is being counted.

## Device Selection (GPU / CPU)

At startup `main.py` asks which device to use:

```
=== CHỌN THIẾT BỊ ===
  1 - GPU
  2 - CPU
```

| Component | Device |
| --------- | ------ |
| YOLOv8 face detector (PyTorch) | **GPU or CPU, controlled by the menu / `--device`** |
| Eye & mouth CNNs (TensorFlow) | CPU if you choose CPU. If you choose GPU, TensorFlow uses a GPU only when your TensorFlow build supports it (native Windows with TensorFlow ≥ 2.11 does **not**, so these CNNs stay on CPU there; use WSL2 for TensorFlow GPU). |
| Haar Cascades (OpenCV) | Always CPU |

If you choose GPU but CUDA is unavailable, the program prints a warning and falls back to CPU automatically. The active YOLO device is shown on the video (`YOLO: cuda:0` or `YOLO: cpu`) and in the startup log.

## Testbed Setup

- Evaluated on a standard laptop with a USB webcam (640×480, 30 FPS).
- Dataset: ~80k eye images, ~5k mouth images, with an 80/20 train-validation split (16,190 eye and 1,023 mouth samples in validation).
- Reports eye-state and yawn training/validation accuracy, and measures end-to-end FPS under typical indoor lighting conditions.

> **Dataset Source:** the eye and mouth image datasets used for training were collected from publicly available sources online (exact origin not tracked). All images are used strictly for academic/educational purposes.

## Results

| Model     | Training Accuracy | Validation Accuracy | Validation Samples |
| --------- | ----------------- | ------------------- | ------------------ |
| Eye CNN   | ~97%              | **86%**             | 16,190             |
| Mouth CNN | ~97%              | **95%**             | 1,023              |

The Mouth CNN demonstrated stronger generalization than the Eye CNN.

Training and validation accuracy curves (see `TrainModel Pic/`):

[![Eye Model Accuracy](https://github.com/Baaotraan0407/drowsiness-detection/raw/main/TrainModel%20Pic/Figure_1.png)](TrainModel%20Pic/Figure_1.png) [![Mouth Model Accuracy](https://github.com/Baaotraan0407/drowsiness-detection/raw/main/TrainModel%20Pic/Figure_2.png)](TrainModel%20Pic/Figure_2.png)

*Training and validation accuracy of the eye-state CNN (left) and yawn CNN (right).*

[![Validation Accuracy Comparison](https://github.com/Baaotraan0407/drowsiness-detection/raw/main/TrainModel%20Pic/Figure_TrainModel.png)](TrainModel%20Pic/Figure_TrainModel.png)

*Validation accuracy comparison of the Eye and Mouth CNNs.*

**System performance:** an average of **15.7 FPS** was measured in real time on a **CPU-only** laptop. FPS with GPU is not reported here; run with option 1 (GPU) and read the `FPS` value shown on the video to compare on your own machine.

## Conclusion

- The system combines modern deep learning (YOLOv8) with efficient classical components (Haar Cascades and lightweight CNNs) and temporal modeling.
- Achieved real-time performance (avg. 15.7 FPS) on a CPU-only laptop, with optional GPU acceleration for the YOLOv8 face detector.
- Demonstrated high classification accuracy for both eye-state (86%) and yawn (95%).
- Fusing PERCLOS, eye-closure duration and yawn counting, plus a recovery mechanism, makes the alert more stable than a single-threshold approach.

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
│   ├── eye_segmentation_unet.h5
│   ├── mouth_detector.h5
│   ├── yolov8-face.pt
│   ├── yolov8n.pt
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

> Note: the `datasets/` and `Models/` folders are **not** included in this repo (see `.gitignore`) due to their large size. See the [Models & Dataset](#models--dataset) section below. The `TrainModel Pic/` folder is kept since the images are small and illustrate the training results.

## Installation

Python 3.9 is recommended (the project was tested with it). Using a virtual environment is strongly advised.

```
git clone https://github.com/Baaotraan0407/drowsiness-detection.git
cd drowsiness-detection

python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Optional: enable GPU for YOLOv8 (NVIDIA only)

`pip install` may install a **CPU-only** build of PyTorch (version string ends with `+cpu`), in which case the GPU option cannot be used. Check your build:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `False`, reinstall PyTorch with CUDA (example for CUDA 12.1):

```
pip uninstall -y torch torchvision torchaudio
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
```

It should now print something like `2.2.1+cu121 True`. If you see NumPy-related errors afterwards, run `pip install "numpy<2"`.

> TensorFlow on native Windows (≥ 2.11) has no GPU support, so the eye/mouth CNNs run on CPU there regardless of the choice. This is fine because YOLOv8 is the heaviest part of the pipeline.

## Models & Dataset

The following files are required inside the `Models/` folder (not included in this repo):

- `yolov8-face.pt` — YOLOv8 face detection model
- `eye_detector.h5` — CNN model for eye-state classification (trained via `Train.py`, output = P(Open))
- `mouth_detector.h5` — CNN model for yawn classification (trained via `Train.py`)
- `haarcascade_eye.xml`, `haarcascade_mcs_mouth.xml` — OpenCV Haar Cascade classifiers
- `eye_segmentation_unet.h5`, `yolov8n.pt` — auxiliary models *(not used by `main.py`)*

`main.py` looks for the models folder in this order:

1. the `--model-dir` command-line option
2. the `MODEL_DIR` environment variable
3. `./Models`, then `./models` in the current directory

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

The best-performing models will be saved to the `models/` folder.

## Usage

```
python main.py                 # shows the device menu: 1 = GPU, 2 = CPU
python main.py --device gpu    # skip the menu, use GPU for YOLOv8
python main.py --device cpu    # skip the menu, use CPU for everything
```

- Press **q** in the video window to quit.
- The video shows: face bounding box, eye state (with P(Open) and EMA), mouth state (with MAR, P(Yawn) and temporal features), PERCLOS, yawns in the last 60 s, overall status (**AWAKE / DROWSY**), the score, FPS and the YOLO device. A red border and a beep indicate **DROWSY**.
- The device menu is read from the terminal. If you run the script from an environment without an interactive terminal, pass `--device gpu` or `--device cpu` instead.

### Common options

| Option | Description |
| ------ | ----------- |
| `--device {gpu,cpu}` | Skip the device menu and choose GPU or CPU directly |
| `--yolo-device` | Fine-grained YOLO device: `auto`, `cpu`, `cuda`, `cuda:0`, ... (overridden by the menu / `--device`) |
| `--model-dir PATH` | Folder containing the models |
| `--camera N` | Camera index (default 0) |
| `--width N` | Resize frame width for speed (default 640, 0 = keep original) |
| `--no-audio` | Disable the audio beep |
| `--save-csv FILE` | Save a per-frame time series (PERCLOS, eye/mouth state, probabilities, score, status) to CSV |
| `--show-eye-crop`, `--show-mouth-crop` | Show the crops that are fed into the CNNs (for debugging) |
| `--use-gaze-guard` | Enable the gaze-down guard |
| `--eye-open-prob-thr`, `--eye-closed-prob-thr` | Hysteresis thresholds for P(Open) (default 0.70 / 0.40) |
| `--eye-closed-debounce SEC` | Time the eye must stay closed to be confirmed (default 0.15) |
| `--detect-interval N` | Run YOLOv8 every N frames (default 10) |
| `--yolo-conf`, `--yolo-iou` | YOLOv8 confidence / NMS IoU thresholds (default 0.50 / 0.45) |
| `--eye-every N`, `--mouth-every N` | Run the eye / mouth pipeline every N frames (default 1 / 2) |
| `--recovery-duration`, `--recovery-perclos`, `--recovery-yawn-cooldown` | Recovery settings (default 3.0 s / 0.15 / 3.0 s) |
| `--eye-rgb`, `--mouth-rgb`, `--eye-invert` | Diagnostics when the CNN output looks wrong (RGB input, or the eye model outputs P(Closed)) |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Logging verbosity |

Run `python main.py --help` for the full list.

## Troubleshooting

| Problem | Fix |
| ------- | --- |
| `YOLO device selected: CPU` even though you chose GPU | PyTorch is a CPU build. See [Optional: enable GPU for YOLOv8](#optional-enable-gpu-for-yolov8-nvidia-only). |
| `Torch not compiled with CUDA enabled` | Same as above; reinstall PyTorch with a CUDA index URL. |
| `Device (TensorFlow - eye/mouth CNN): CPU` | Expected on native Windows. TensorFlow GPU needs WSL2 or Linux. |
| `FileNotFoundError: ... model not found` | Check the `Models/` folder, or pass `--model-dir` / set `MODEL_DIR`. |
| `Could not open camera` | Try another index, e.g. `--camera 1`. |
| NumPy errors such as `_ARRAY_API not found` | `pip install "numpy<2"`. |
| The program seems stuck right after starting | It is waiting for the device menu in the terminal. Type `1` or `2` and press Enter. |

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8 (PyTorch)
- TensorFlow / Keras
- NumPy, Matplotlib