"""
STEP 1: Collect hand landmark data for each SgSL letter.

How to use:
- Run this script: python collect_data.py
- It will automatically download the hand detection model (first run only, ~26MB)
- Press a letter key (A-Z) to start collecting data for that letter
- Hold up the sign for that letter while the script captures 100 samples
- Repeat for every letter you want to train
- Data is saved to a folder called 'data/'
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request

# ── Download model if not present ──────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand detection model (one-time, ~26MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✓ Model downloaded.")

# ── Setup ──────────────────────────────────────────────────────────────────
DATA_DIR           = "data"
SAMPLES_PER_LETTER = 100
os.makedirs(DATA_DIR, exist_ok=True)

# New MediaPipe Tasks API
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    running_mode=mp_vision.RunningMode.IMAGE
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# Hand skeleton connections for drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ── Helpers ────────────────────────────────────────────────────────────────
def get_landmarks(frame):
    """Detect hand and return 42 normalised (x, y) values, or None."""
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, None

    hand  = result.hand_landmarks[0]
    raw_x = [lm.x for lm in hand]
    raw_y = [lm.y for lm in hand]

    min_x, min_y = min(raw_x), min(raw_y)
    norm = []
    for x, y in zip(raw_x, raw_y):
        norm.append(x - min_x)
        norm.append(y - min_y)

    return norm, hand


def draw_hand(frame, landmarks):
    """Draw hand skeleton on frame."""
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


# ── Main loop ──────────────────────────────────────────────────────────────
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # lower resolution = faster
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Warm up the camera — some webcams need a moment to start
print("Starting camera, please wait...")
for _ in range(30):
    cap.read()
    time.sleep(0.05)

print("Camera ready. Press a letter key (A-Z) to collect data for that letter.")
print("Press Q to quit.\n")

current_letter  = None
collecting      = False
collected_count = 0
frame_count     = 0

# ── Load existing data if it exists ────────────────────────────────────────
X_path = os.path.join(DATA_DIR, "X.npy")
y_path = os.path.join(DATA_DIR, "y.npy")

if os.path.exists(X_path) and os.path.exists(y_path):
    X_existing = np.load(X_path)
    y_existing = np.load(y_path)
    all_data = list(zip(X_existing.tolist(), y_existing.tolist()))
    existing_letters = sorted(set(y_existing))
    print(f"Loaded existing data: {existing_letters} ({len(X_existing)} samples)")
    print("Tip: To REDO a letter, press DELETE then that letter key.")
else:
    all_data = []
    print("No existing data found — starting fresh.")

delete_mode = False   # when True, next letter press will delete that letter's data

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from webcam. Check your camera.")
        break

    frame      = cv2.flip(frame, 1)
    frame_count += 1

    # Only run hand detection every 3 frames — reduces CPU load
    if frame_count % 3 == 0:
        landmarks, hand_lm = get_landmarks(frame)
    else:
        landmarks, hand_lm = None, None

    if hand_lm:
        draw_hand(frame, hand_lm)

    if collecting and current_letter:
        status = f"Collecting '{current_letter}': {collected_count}/{SAMPLES_PER_LETTER}"
        cv2.putText(frame, status, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        if landmarks:
            all_data.append((landmarks, current_letter))
            collected_count += 1

        if collected_count >= SAMPLES_PER_LETTER:
            print(f"  ✓ Done collecting '{current_letter}'")
            collecting      = False
            collected_count = 0
            current_letter  = None
    else:
        if delete_mode:
            cv2.putText(frame, "DELETE MODE: Press letter to redo it.", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "A-Z: collect  DEL+letter: redo  ESC: quit", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    letters_done = sorted(set(label for _, label in all_data))
    info = "Collected: " + (", ".join(letters_done) if letters_done else "none")
    cv2.putText(frame, info, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("SgSL Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key to quit
        break
    elif key == 46:  # DELETE key — next letter press will redo that letter
        delete_mode = True
        print("DELETE MODE: Press a letter to remove all its samples and recollect.")
    elif key < 128 and chr(key).isalpha():
        letter = chr(key).upper()
        if delete_mode:
            before = len(all_data)
            all_data = [(x, y) for x, y in all_data if y != letter]
            removed = before - len(all_data)
            print(f"Deleted {removed} samples for '{letter}'. Now recollecting — show the sign!")
            delete_mode = False
        else:
            print(f"Starting collection for letter '{letter}' — show the sign now!")
        current_letter  = letter
        collecting      = True
        collected_count = 0

# ── Save ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
detector.close()

if all_data:
    X = np.array([item[0] for item in all_data])
    y = np.array([item[1] for item in all_data])
    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)
    print(f"\n✓ Saved {len(X)} total samples for letters: {sorted(set(y))}")
    print("  Next step → run:  python train_model.py")
else:
    print("No data collected.")
