"""
SgSL Real-Time Translator v2 — with text-to-speech, save to file, sentence history.

- Hold a sign steady for 1s → letter added automatically
- Lower hand for 1s → space added, word is spoken aloud
- Lower hand for 3s → sentence saved to history and cleared
- Press ESC to quit → full session saved to a text file

Install extra library before running:
    python -m pip install pyttsx3
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import urllib.request
import threading
import datetime
from collections import deque

# ── Text-to-speech settings — adjust these to change the voice ────────────
TTS_RATE       = 1      # speed:  -10 (slowest) → 0 (normal) → 10 (fastest)
TTS_VOLUME     = 100    # volume: 0 (silent) → 100 (loudest)
TTS_VOICE      = 0      # voice index — run the app once to see available voices printed below

# ── Text-to-speech setup (fast, uses Windows SAPI directly) ───────────────
try:
    import win32com.client
    _speaker = win32com.client.Dispatch("SAPI.SpVoice")

    # Print available voices so you can choose
    voices = _speaker.GetVoices()
    print("\nAvailable voices:")
    for i in range(voices.Count):
        print(f"  [{i}] {voices.Item(i).GetDescription()}")
    print()

    # Apply settings
    _speaker.Voice   = voices.Item(TTS_VOICE)
    _speaker.Rate    = TTS_RATE
    _speaker.Volume  = TTS_VOLUME
    TTS_AVAILABLE = True
    print(f"Text-to-speech ready. Voice [{TTS_VOICE}], Rate {TTS_RATE}, Volume {TTS_VOLUME}.")
except ImportError:
    TTS_AVAILABLE = False
    print("pywin32 not installed. Run: python -m pip install pywin32")

def speak(text):
    """Speak text in a background thread — near-instant, no process startup delay."""
    if TTS_AVAILABLE and text.strip():
        def _speak():
            _speaker.Speak(text)
        threading.Thread(target=_speak, daemon=True).start()

# ── Download model if not present ──────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand detection model (~26MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# ── Load trained model ─────────────────────────────────────────────────────
PICKLE_PATH = os.path.join("model", "sgsl_model.pkl")
if not os.path.exists(PICKLE_PATH):
    print("ERROR: Run train_model.py first.")
    exit()

with open(PICKLE_PATH, "rb") as f:
    saved = pickle.load(f)
model = saved["model"]
le    = saved["label_encoder"]
print(f"Model loaded. Letters: {list(le.classes_)}\n")

# ── MediaPipe setup ────────────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

detector = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE
    )
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ── Settings ───────────────────────────────────────────────────────────────
HOLD_TO_ADD    = 1.0
SPACE_AFTER    = 1.0
CLEAR_AFTER    = 3.0
MIN_CONFIDENCE = 0.60
MAX_HISTORY    = 4      # number of past sentences to show on screen

# ── Helpers ────────────────────────────────────────────────────────────────
def get_landmarks(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return None, None
    hand  = result.hand_landmarks[0]
    xs    = [lm.x for lm in hand]
    ys    = [lm.y for lm in hand]
    mx, my = min(xs), min(ys)
    norm  = []
    for x, y in zip(xs, ys):
        norm += [x - mx, y - my]
    return norm, hand

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)

def draw_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def save_session(history_log):
    """Save all sentences from this session to a timestamped text file."""
    os.makedirs("sessions", exist_ok=True)
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath   = os.path.join("sessions", f"session_{timestamp}.txt")
    with open(filepath, "w") as f:
        f.write(f"SgSL Translator Session — {timestamp}\n")
        f.write("=" * 40 + "\n\n")
        for i, sentence in enumerate(history_log, 1):
            f.write(f"{i}. {sentence}\n")
    print(f"\nSession saved to {filepath}")
    return filepath

# ── State ──────────────────────────────────────────────────────────────────
sentence          = []
history           = deque(maxlen=12)
sentence_history  = []    # completed sentences shown on screen

# Hold-to-add
hold_letter       = None
hold_start        = None
last_added        = None

# No-hand timing
no_hand_since     = None
space_done        = False

frame_count    = 0
last_landmarks = None
last_hand_lm   = None

# ── Camera ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Warming up camera...")
for _ in range(20):
    cap.read()
    time.sleep(0.05)
print("Ready. Hold a sign steady to add it. ESC to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    now   = time.time()
    frame_count += 1

    # Run detection every 3 frames, reuse last result on skipped frames
    if frame_count % 3 == 0:
        landmarks, hand_lm = get_landmarks(frame)
        last_landmarks = landmarks
        last_hand_lm   = hand_lm
    else:
        landmarks = last_landmarks
        hand_lm   = last_hand_lm

    # ── Hand present ──────────────────────────────────────────────────────
    if landmarks:
        draw_hand(frame, hand_lm)

        proba  = model.predict_proba(np.array(landmarks).reshape(1, -1))[0]
        idx    = np.argmax(proba)
        conf   = proba[idx]
        letter = le.classes_[idx] if conf >= MIN_CONFIDENCE else None

        if letter:
            history.append(letter)
            stable = max(set(history), key=history.count)
            current = stable if history.count(stable) >= len(history) * 0.6 else None
        else:
            current = None
            history.clear()

        no_hand_since = None
        space_done    = False

        # ── Hold logic ────────────────────────────────────────────────────
        if current:
            if current != hold_letter:
                hold_letter = current
                hold_start  = now

            hold_secs = now - hold_start
            progress  = min(hold_secs / HOLD_TO_ADD, 1.0)

            if hold_secs >= HOLD_TO_ADD and current != last_added:
                sentence.append(current)
                last_added = current
                hold_start = now
                print(f"Added '{current}' -> {''.join(sentence)}")
        else:
            hold_letter = None
            hold_start  = None
            progress    = 0.0
            current     = ""
            conf        = 0.0

        # ── Bottom UI ─────────────────────────────────────────────────────
        draw_rect(frame, 0, h-130, w, h, (20, 20, 20))

        if current:
            bar_col = (0, 200, 0) if conf > 0.8 else (0, 160, 210)
            cv2.putText(frame, current, (w//2 - 30, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, bar_col, 5)
            cv2.rectangle(frame, (10, h-128), (10+int((w-20)*conf), h-113), (80,180,80), -1)
            cv2.rectangle(frame, (10, h-128), (w-10, h-113), (80,80,80), 1)
            cv2.putText(frame, f"Confidence: {conf*100:.0f}%",
                        (14, h-115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
            p_col = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
            cv2.rectangle(frame, (10, h-111), (10+int((w-20)*progress), h-96), p_col, -1)
            cv2.rectangle(frame, (10, h-111), (w-10, h-96), (80,80,80), 1)
            label = "ADDED!" if progress >= 1.0 else f"Hold steady... {hold_secs:.1f}s / {HOLD_TO_ADD}s"
            cv2.putText(frame, label, (14, h-98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
        else:
            cv2.putText(frame, "Sign detected but confidence too low",
                        (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1)

    # ── No hand ───────────────────────────────────────────────────────────
    else:
        history.clear()
        hold_letter = None
        hold_start  = None
        last_added  = None

        if no_hand_since is None:
            no_hand_since = now
        gap = now - no_hand_since

        # Add space only — no speech yet, you may still be signing more words
        if gap >= SPACE_AFTER and not space_done and sentence and sentence[-1] != " ":
            sentence.append(" ")
            space_done = True
            print(f"[space] -> {''.join(sentence)}")

        # Clear + save to history
        if gap >= CLEAR_AFTER and sentence:
            completed = "".join(sentence).strip()
            if completed:
                sentence_history.append(completed)
                if len(sentence_history) > MAX_HISTORY:
                    sentence_history.pop(0)
                speak(completed)   # speak the full sentence on clear
                print(f"[clear] Saved: '{completed}'")
            sentence.clear()
            space_done = False

        draw_rect(frame, 0, h-80, w, h, (20, 20, 20))
        if gap < SPACE_AFTER:
            msg = "No hand — show a sign"
        elif gap < CLEAR_AFTER:
            msg = f"Clearing in {CLEAR_AFTER - gap:.1f}s — raise hand to continue"
        else:
            msg = "Ready — show a sign"
        cv2.putText(frame, msg, (10, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140,140,140), 1)

    # ── Top bar — current sentence ────────────────────────────────────────
    draw_rect(frame, 0, 0, w, 58, (30, 30, 30))
    display = "".join(sentence).strip() if sentence else "Waiting for sign..."
    cv2.putText(frame, display, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    cv2.putText(frame, "Hold 1s=add letter | No hand 1s=space | 3s=speak+clear | ESC=quit+save",
                (10, h-83), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130,130,130), 1)

    # ── History panel — dedicated strip BELOW the camera ─────────────────
    HIST_H = 115
    canvas = np.zeros((h + HIST_H, w, 3), dtype=np.uint8)
    canvas[:h] = frame                              # camera view on top
    canvas[h:] = (28, 28, 28)                       # dark history background
    cv2.line(canvas, (0, h), (w, h), (70, 130, 70), 2)   # green separator
    cv2.putText(canvas, "SENTENCE HISTORY", (10, h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 80), 1)
    if sentence_history:
        recent = list(reversed(sentence_history))[:3]   # newest first, up to 3
        for i, past in enumerate(recent):
            display_past = past if len(past) <= 60 else past[:57] + "..."
            color = (230, 230, 230) if i == 0 else (140, 140, 140)
            prefix = ">" if i == 0 else " "
            cv2.putText(canvas, f"{prefix} {display_past}",
                        (14, h + 42 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1)
    else:
        cv2.putText(canvas, "Completed sentences appear here",
                    (14, h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (70, 70, 70), 1)

    cv2.imshow("SgSL Translator v2", canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ── Cleanup and save session ───────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
detector.close()

# Add any remaining sentence to history before saving
if sentence:
    remaining = "".join(sentence).strip()
    if remaining:
        sentence_history.append(remaining)

if sentence_history:
    filepath = save_session(sentence_history)
    print(f"Session saved to: {filepath}")
else:
    print("No sentences to save.")

print(f"\nFinal: {''.join(sentence).strip()}")
