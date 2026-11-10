<p align="center">
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/English-1f6feb?style=for-the-badge"></a>
  <a href="README.ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-555555?style=for-the-badge"></a>
</p>

# Invoker CV 

### Play Invoker in Dota 2 with hand gestures


<div align="center">

https://github.com/user-attachments/assets/9d406dd6-4915-4d4a-96cd-1b9d0fe17908

Show a gesture to your webcam: the neural network finds your hands, recognises the pose<br>
and presses Q / W / E / R / D / F for you. No macros: one gesture, one key.

<br>

<img src="https://img.shields.io/badge/Dota_2-Invoker-B22222?style=for-the-badge&logo=steam&logoColor=white" alt="Dota 2 Invoker">

<br><br>

<img src="https://img.shields.io/badge/gestures-6_+_none-orange?style=flat-square" alt="gestures">
<img src="https://img.shields.io/badge/Python-3.10–3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/OpenCV-4.11-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/MediaPipe-Hands-0097A7?style=flat-square&logo=google&logoColor=white" alt="MediaPipe">
<img src="https://img.shields.io/badge/scikit--learn-MLP-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn">
<img src="https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Windows-pydirectinput-0078D6?style=flat-square&logo=windows&logoColor=white" alt="Windows">

</div>

---

## Gestures

| Gesture   | Key     | Hands | How to show it                                                  | Auto-repeat |
|-----------|:-------:|:-----:|-----------------------------------------------------------------|:-----------:|
| `quas`    | `Q`     | 2     | Triangle: index fingers touch at the top, thumbs at the bottom  | 450 ms      |
| `wex`     | `W`     | 2     | Two fists side by side, both thumbs up                          | 450 ms      |
| `exort`   | `E`     | 2     | Index fingers horizontal, tips touching, thumbs up              | 450 ms      |
| `invoke`  | `R`     | 2     | Palms together, fingers up ("roof")                             | —           |
| `spell_d` | `D`     | 1     | One hand, index finger up                                       | —           |
| `spell_f` | `F`     | 1     | One hand, two fingers up (V)                                    | —           |
| `none`    | —       | —     | Hands down / random movements / transitions between gestures    | —           |

While you hold an orb (`quas`/`wex`/`exort`), the key repeats every `repeat_ms`. To press the same gesture again without the repeat, lower your hands or switch gestures.

## How it works

1. **`hands.py`**: MediaPipe Hands finds up to two hands, and each frame becomes a 156-feature vector:
   - left and right hand shape (2 × 63): landmarks relative to the wrist, normalised by palm size, so they do not depend on hand position in the frame or distance to the camera;
   - right wrist position relative to the left one (3);
   - pairwise distances between the fingertips of both hands (25), to catch "fingers touching";
   - "hand found" flags (2).
2. **`train.py`**: `StandardScaler` + `MLPClassifier(128, 64)` from scikit-learn. Trains in seconds on CPU. Augmentations: noise, scaling, ±10° hand rotation, swapping and mirroring of one-handed frames (MediaPipe sometimes confuses left and right hands).
3. **`controller.py`**: a gesture fires when it is held for `stable_frames` consecutive frames with confidence ≥ `threshold`. Two-handed gestures do not fire if only one hand is in the frame. Key presses go through `pydirectinput` (SendInput with scan codes, which the game sees).

## Installation

You need Python 3.10–3.12 (`mediapipe 0.10.21` and `numpy 1.26` are built for it) and a webcam.

```bash
git clone https://github.com/CONt12006/dota_invoker_cv_gameplay.git
cd dota_invoker_cv_gameplay
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

> Key presses are sent to the game via `pydirectinput`, which works on **Windows** only. On Linux the controller can be run with `--dry-run` (recognition without key presses).

## Quick start

The repository already contains recorded gestures (`data/dota_gestures/`) and a trained model (`models/dota_gestures.joblib`), so you can run it right away:

```bash
python controller.py --dry-run   # check recognition, presses nothing
python controller.py             # live mode, starts PAUSED
```

- **F8** (global) or **P** in the window: enable / pause.
- **Q** / **Esc** in the window: quit.
- `--armed`: start already enabled.
- `--camera N`: choose another camera.

In Dota, set the **borderless window** mode so the controller window stays on top of the game.

## Your own gestures

The model is trained on my hands and my camera, so it works best for you if you record your own.

### 1. Recording

```bash
python record.py              # 6-second sessions by default
python record.py --seconds 10
```

| Key         | Action                                        |
|-------------|-----------------------------------------------|
| `0`–`9`     | select a gesture (`0` is `none`)              |
| `n` / `p`   | next / previous gesture                       |
| `Space`     | record a session (1 s countdown + recording)  |
| `u`         | delete the last session                       |
| `q` / `Esc` | quit                                          |

Tips:

- while recording, hold the gesture and vary it slightly: rotate the hand, move closer/farther, higher/lower;
- for `none`, lower your hands, scratch your nose, show transitions between gestures: this teaches the model **not** to press keys by accident;
- aim for 300+ frames and 3+ sessions per gesture.

Each session is saved to `data/dota_gestures/<gesture>/<timestamp>.npy` (an array of `frames × 156`).

### 2. Training

```bash
python train.py
```

The script prints a quality report on **unseen sessions** (`GroupKFold` by recording: adjacent frames of one session are nearly identical and would inflate accuracy), a confusion matrix, and saves the model to `models/dota_gestures.joblib`.

### 3. Running

```bash
python controller.py
```

## Configuration: `config.json`

```json
{
  "camera": 0,
  "threshold": 0.80,
  "stable_frames": 4,
  "toggle_hotkey": "<f8>",
  "gestures": {
    "quas": {"key": "q", "hands": 2, "repeat_ms": 450, "hint": "..."}
  }
}
```

| Field           | What it does                                                                  |
|-----------------|-------------------------------------------------------------------------------|
| `camera`        | webcam index                                                                  |
| `threshold`     | minimum model confidence for a frame to count                                 |
| `stable_frames` | how many consecutive frames a gesture must be held (more is safer but slower) |
| `toggle_hotkey` | global pause hotkey (`pynput` format)                                         |
| `key`           | which key to press                                                            |
| `hands`         | how many hands the gesture needs                                              |
| `repeat_ms`     | auto-repeat interval while held, `0` for no repeat                            |
| `hint`          | hint shown in the recording window                                            |

To add a new gesture, add it to `gestures`, record it with `record.py` and retrain with `train.py`.

## Structure

```
.
├── hands.py          # hand detection, features, augmentations, drawing
├── record.py         # dataset recording from the webcam
├── train.py          # MLP training and validation
├── controller.py     # gesture -> key press
├── config.json       # gestures, keys, thresholds
├── data/dota_gestures/   # recorded sessions (.npy)
└── models/dota_gestures.joblib
```

## Stack

Python · OpenCV · MediaPipe Hands · NumPy · scikit-learn · pydirectinput · pynput
