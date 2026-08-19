# 🤟 Real-Time SgSL Translator

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-green)
![OpenCV](https://img.shields.io/badge/OpenCV-latest-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![Status](https://img.shields.io/badge/Status-Working%20Prototype-brightgreen)

A real-time **Singapore Sign Language (SgSL)** alphabet translator built with Python, MediaPipe, and machine learning. Shows a sign to the webcam, hold it for 1 second, and it automatically spells out words — no keyboard needed.

---

## Demo

> Show hand sign → hold 1 second → letter added automatically
> Lower hand 1 second → space
> Lower hand 3 seconds → clear

---

## Features

- Real-time hand landmark detection using Google MediaPipe
- Custom-trained Random Forest classifier for SgSL alphabet (A-Z)
- Automatic word building — no keyboard required
- Live confidence score and hold progress bar on screen
- Data collection and model training scripts included

---

## Getting Started

### Prerequisites

- Python 3.11
- Windows / Mac / Linux
- Webcam

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/sgsl-translator.git
cd sgsl-translator

# Create virtual environment
py -3.11 -m venv sgsl_env
sgsl_env\Scripts\activate   # Windows
source sgsl_env/bin/activate  # Mac/Linux

# Install dependencies
python -m pip install mediapipe==0.10.35 opencv-python scikit-learn numpy
```

### Usage

Run the scripts in order:

```bash
# Step 1 — Collect your sign data
python collect_data.py

# Step 2 — Train the model
python train_model.py

# Step 3 — Run the live translator
python app.py
```

---

## How to Collect Data

1. Run `collect_data.py`
2. Click on the webcam window
3. Press a letter key (A-Z) to start collecting for that letter
4. Hold your SgSL sign steady — 100 samples are captured automatically
5. Press ESC when done — data saved to `data/` folder

To redo a letter: press `DELETE` then the letter key.

---

## Project Structure

```
sgsl-translator/
├── collect_data.py       # Step 1: collect hand landmark data
├── train_model.py        # Step 2: train the classifier
├── app.py                # Step 3: run the live translator
├── test_camera.py        # Camera and hand detection diagnostic
├── hand_landmarker.task  # MediaPipe model (auto-downloaded)
├── data/
│   ├── X.npy             # Landmark feature data
│   └── y.npy             # Labels
└── model/
    └── sgsl_model.pkl    # Trained classifier
```

---

## How It Works

### 1. Hand Detection
MediaPipe's Hand Landmarker detects 21 key points on the hand per frame, producing 42 normalised (x, y) coordinates. Normalisation removes position bias so only hand shape matters.

### 2. Classification
A Random Forest Classifier trained on self-collected landmark data maps hand shapes to SgSL letters. Training takes seconds and achieves high accuracy on clean data.

### 3. Auto-add Logic
The app buffers the last 12 predictions and checks for stability. When the same letter is consistently predicted for 1 second, it is added to the word automatically.

---

## Challenges

| Challenge | Solution |
|---|---|
| MediaPipe `mp.solutions` deprecated | Rewrote using new Tasks API |
| Python 3.14 incompatible | Used Python 3.11 with venv |
| Hold timer resetting every frame | Cached last detection result across skipped frames |
| App lagging on CPU | Run detection every 3rd frame only |
| Signs not recognised | Recollected training data in consistent lighting |

---

## Limitations

- Alphabet only — full word signs not yet supported
- Trained on one person's signing style
- Single hand tracking only
- Static signs only — motion-based signs not supported

---

## Roadmap

- [ ] Full SgSL word recognition using LSTM + video sequences
- [ ] Two-hand and body pose support via MediaPipe Holistic
- [ ] Text-to-speech output
- [ ] Multi-signer training data
- [ ] Mobile app version
- [ ] Integration with NTU SgSL SignBank (pending SADeaf permission)

---

## References

- [NTU SgSL SignBank](https://blogs.ntu.edu.sg/sgslsignbank/)
- [Google MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Singapore Association for the Deaf](https://sadeaf.org.sg)

---

## Acknowledgements

Inspired by the Deaf and hard-of-hearing community in Singapore. Built as a personal learning project to explore applied AI and computer vision beyond data analysis.

---

*If you find this useful or want to collaborate, feel free to open an issue or reach out.*
