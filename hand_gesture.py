# TechVidvan Hand Gesture Recognizer

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# Check if the model file exists
model_path = 'mp_hand_gesture.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' was not found.")

# Load the gesture recognizer model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Load class names
gesture_names_path = 'gesture.names'
if not os.path.exists(gesture_names_path):
    raise FileNotFoundError(f"The gesture names file '{gesture_names_path}' was not found.")

with open(gesture_names_path, 'r') as f:
    classNames = f.read().strip().split('\n')
print("Class names loaded:", classNames)

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    className = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.extend([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Flatten and predict gesture
            landmarks = np.array(landmarks).reshape(1, -1)
            try:
                prediction = model.predict(landmarks, verbose=0)
                classID = np.argmax(prediction)
                className = classNames[classID]
            except Exception as e:
                print(f"Prediction error: {e}")

    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
