"""
STEP 2: Train a sign language classifier on the data you collected.

How to use:
- Make sure you have run collect_data.py first
- Run: python train_model.py
- This saves a trained model to 'model/sgsl_model.pkl'
"""

import numpy as np
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Load data ──────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

X_path = os.path.join(DATA_DIR, "X.npy")
y_path = os.path.join(DATA_DIR, "y.npy")

if not os.path.exists(X_path) or not os.path.exists(y_path):
    print("ERROR: Data files not found.")
    print("Please run collect_data.py first to collect your sign data.")
    exit()

X = np.load(X_path)
y = np.load(y_path)

print(f"Loaded {len(X)} samples for letters: {sorted(set(y))}")

# ── Encode labels (A, B, C → 0, 1, 2) ────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── Split into train / test sets ───────────────────────────────────────────
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── Train the model ────────────────────────────────────────────────────────
# RandomForest is a good starting model — fast, accurate, beginner-friendly
print("\nTraining model... (this takes a few seconds)")

model = RandomForestClassifier(
    n_estimators=100,   # number of decision trees
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ Model accuracy: {accuracy * 100:.1f}%")
print("\nDetailed results:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

if accuracy < 0.80:
    print("⚠  Accuracy is below 80%. Tips to improve:")
    print("   • Collect more samples (increase SAMPLES_PER_LETTER in collect_data.py)")
    print("   • Make sure your hand is clearly visible when collecting")
    print("   • Try different lighting conditions")

# ── Save model + label encoder ─────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "sgsl_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "label_encoder": le}, f)

print(f"\n✓ Model saved to {model_path}")
print("  Next step → run:  python app.py")
