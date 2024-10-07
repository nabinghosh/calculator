import sys
import os
# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import mediapipe as mp
import torch
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from src.gemini_api import calculate_expression



# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Drawing parameters
drawing = False
erase_mode = False
points = []
canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White canvas

# Load the fine-tuned model
feature_extractor = ViTFeatureExtractor.from_pretrained("./fine-tuned-model")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")
model = VisionEncoderDecoderModel.from_pretrained("./fine-tuned-model")

# Function to recognize gestures
def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]

    # Gesture detection: Calculate distances between thumb and index finger
    thumb_index_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    if thumb_index_dist < 0.05:  # Pinch gesture (solve equation)
        return "solve"
    if index_tip.y < middle_tip.y and middle_tip.y < landmarks[16].y:  # Fist gesture (clear screen)
        return "clear"
    return "write"

# Function to draw on the canvas
def draw_on_canvas(points, canvas):
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(canvas, points[i - 1], points[i], (0, 0, 0), 5)
    return canvas

# Function to preprocess the canvas image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224))
    normalized = resized / 255.0
    preprocessed = np.expand_dims(normalized, axis=0)
    preprocessed = np.expand_dims(preprocessed, axis=-1)  # Add channel dimension
    return preprocessed

# Function to recognize handwritten equations
def recognize_equation(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_equation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_equation

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            gesture = detect_gesture(landmarks)

            if gesture == "write":
                index_tip = landmarks[8]
                x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                points.append((x, y))
            elif gesture == "clear":
                points = []
                canvas.fill(255)
            elif gesture == "solve":
                # Preprocess the canvas and recognize the equation
                preprocessed_image = preprocess_image(canvas)
                try:
                    equation = recognize_equation(preprocessed_image)
                    result = calculate_expression(equation)
                    cv2.putText(frame, f'Result: {result}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error recognizing equation: {e}")

    # Draw on the canvas
    canvas = draw_on_canvas(points, canvas)

    # Display the canvas and frame
    cv2.imshow('Canvas', canvas)
    cv2.imshow('AI Calculator', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()