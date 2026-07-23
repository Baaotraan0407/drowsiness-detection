<<<<<<< HEAD
# Real-Time Driver Drowsiness Detection using YOLOv8, Haar Cascades, and CNN Classifiers

A real-time driver drowsiness detection system that combines **YOLOv8** for face detection, **Haar Cascades** for eye/mouth region extraction, and **lightweight CNNs** for eye-state and yawn classification. The system fuses temporal features (MAR, PERCLOS) with a gaze-down guard to produce a stable drowsiness score, running in real time on CPU-only hardware.

**Authors:** Lam Bao Tran (1113540), Baitikova Bermet (1113531)
**Advisor:** Prof. Naeem Ul Islam
**International Bachelor Program in Informatics (IBPI)**

## Abstract

Driver fatigue is a major factor in road accidents. This project proposes a real-time, vision-based system that uses YOLOv8 for robust face detection, Haar Cascades for region extraction, and lightweight CNNs for eye-state and yawn classification. The core innovation is fusing temporal features (MAR, PERCLOS) and a gaze-down guard for a stable, real-time drowsiness score on CPU-only hardware.

## Model Architecture & Methodology

The pipeline is a sequence of modular components that runs in real time:

```
Input Frame (Webcam)
        │
        ▼
  YOLOv8 Face Detector
        │
   ┌────┴────┐
   ▼         ▼
Haar Cascade   Haar Cascade
 Eye ROI        Mouth ROI
   │              │
   ▼              ▼
 Eye CNN       Mouth CNN
Open/Closed    Yawn/Non-yawn
   │              │
   └──────┬───────┘
          ▼
Temporal Features (MAR, ZCR, Peaks,
     Plateau, Above-threshold Area)
          │
          ▼
   PERCLOS + Gaze-down Guard
       (Drowsiness Score)
          │
          ▼
   Driver Alert (Visual / Audio)
```

- **YOLOv8 Face Detector**: detects the driver's face in each webcam frame.
- **Haar Cascades**: extract the eye ROI and mouth ROI from the detected face.
- **Two lightweight CNNs**: classify eye-state (open/closed) and yawn (yawn/non-yawn).
- **Temporal module**: aggregates frame-level predictions, computing MAR-based features (zero-crossing rate, peaks, plateau duration, above-threshold area) along with PERCLOS and a gaze-down guard, producing a stable drowsiness score that triggers a visual/audio alert in real time on CPU-only hardware.

## Testbed Setup

- Evaluated on a standard laptop with a USB webcam (640×480, 30 FPS).
- YOLOv8, Haar Cascades, and both CNNs all run on CPU.
- Dataset: ~80k eye images, ~5k mouth images, with an 80/20 train-validation split (16,190 eye and 1,023 mouth samples in validation).
- Reports eye-state and yawn training/validation accuracy, and measures end-to-end FPS under typical indoor lighting conditions.

## Results

| Model     | Training Accuracy | Validation Accuracy | Validation Samples |
|-----------|--------------------|-----------------------|-----------------------|
| Eye CNN   | ~97%               | **86%**               | 16,190                |
| Mouth CNN | ~97%               | **95%**                | 1,023                 |

The Mouth CNN demonstrated stronger generalization than the Eye CNN.

Training and validation accuracy curves (see `TrainModel Pic/`):

<p align="center">
  <img src="TrainModel%20Pic/Figure_1.png" width="45%" alt="Eye Model Accuracy">
  <img src="TrainModel%20Pic/Figure_2.png" width="45%" alt="Mouth Model Accuracy">
</p>
<p align="center"><i>Training and validation accuracy of the eye-state CNN (left) and yawn CNN (right).</i></p>

<p align="center">
  <img src="TrainModel%20Pic/Figure_TrainModel.png" width="60%" alt="Validation Accuracy Comparison">
</p>
<p align="center"><i>Validation accuracy comparison of the Eye and Mouth CNNs.</i></p>

**System performance:** achieved an average of **15.7 FPS** running in real time on a CPU-only laptop.

## Conclusion

- The proposed system successfully combines modern deep learning (YOLOv8) with efficient classical components (Haar Cascades and lightweight CNNs) and temporal modeling.
- Achieved real-time performance (avg. 15.7 FPS) on a CPU-only laptop.
- Demonstrated high classification accuracy for both eye-state (86%) and yawn (95%).
- The temporal features and PERCLOS-based decision module enable robust and reliable fatigue prediction.

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

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Models & Dataset

The following files are required inside the `Models/` folder (not included in this repo):

- `yolov8-face.pt` — YOLOv8 face detection model
- `eye_detector.h5` — CNN model for eye-state classification (trained via `Train.py`)
- `mouth_detector.h5` — CNN model for yawn classification (trained via `Train.py`)
- `haarcascade_eye.xml`, `haarcascade_mcs_mouth.xml` — OpenCV Haar Cascade classifiers
- `eye_segmentation_unet.h5`, `yolov8n.pt` — auxiliary models *(add a description if used in your pipeline)*

Place these files in a `Models/` folder, then set the `MODEL_DIR` environment variable to point to it:

```bash
# Windows (PowerShell)
$env:MODEL_DIR="D:\path\to\Models"

# Windows (cmd)
set MODEL_DIR=D:\path\to\Models

# macOS / Linux
export MODEL_DIR=/path/to/Models
```

If not set, the code falls back to a hardcoded path in `main.py` — you should either update that path or always set `MODEL_DIR` when running on a different machine.

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

```bash
python Train.py
```

The best-performing models will be saved to the `models/` folder.

## Usage

```bash
python main.py
```

- Press **q** to quit.
- The system displays the webcam feed with: face bounding box, eye state, mouth state, and overall status (AWAKE / DROWSY).

## Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8
- TensorFlow / Keras
- NumPy, Matplotlib
=======
# drowsiness-detection
Real-time driver drowsiness detection system using YOLOv8, Haar Cascades, and lightweight CNNs to classify eye-state and yawn, with temporal (MAR/PERCLOS) fusion for stable drowsiness alerts.
>>>>>>> 9b59d310467d666eda0f9185a88dfed4be189fd0
