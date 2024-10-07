import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224))
    normalized = resized / 255.0
    preprocessed = np.expand_dims(normalized, axis=0)
    return preprocessed