import os
import sys
import cv2
import numpy as np
import pickle
from skimage.feature import hog

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = os.path.join("models", "emotion_pipeline.pkl")

LABEL_MAP = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

HOG_PARAMS = dict(
    orientations=8,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True,
)

# -----------------------------
# Load pipeline
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at: {MODEL_PATH}")
    print("Make sure you ran the notebook cell to create models/emotion_pipeline.pkl")
    sys.exit(1)

with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

# -----------------------------
# Camera & face detector
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open camera. Try a different index: 1 or 2.")
    print("Example: change to cv2.VideoCapture(1)")
    sys.exit(1)

# Use OpenCV’s built-in Haar cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print("[ERROR] Haar cascade not found at:", cascade_path)
    sys.exit(1)
face_cascade = cv2.CascadeClassifier(cascade_path)

print("Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Camera frame read failed.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces (tune params if needed)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48)
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))

        # HOG feature (same as training)
        feat = hog(roi, **HOG_PARAMS).reshape(1, -1)

        # scale -> pca -> svm (inside pipeline)
        pred = int(pipeline.predict(feat)[0])
        label = LABEL_MAP.get(pred, str(pred))

        # draw UI
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Emotion Recognition (SVM)", frame)

    # Check if window was closed or 'q' pressed
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        print("Quitting...")
        break
    if cv2.getWindowProperty("Emotion Recognition (SVM)", cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed.")
        break


cap.release()
cv2.destroyAllWindows()
