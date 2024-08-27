import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model  # Change according to your model type
from tensorflow.keras.preprocessing import image  # If using Keras for preprocessing

# Load your corrosion detection model
model = load_model('corrosion_detection_model.h5')  # Adjust path and model type as needed

def apply_corrosion_detection(img):
    # Preprocess the image for your model
    img = cv2.resize(img, (224, 224))  # Adjust size based on model input
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize if required

    # Predict using the model
    prediction = model.predict(img)
    return prediction

def capture_and_process():
    screen_capture = cv2.VideoCapture(0)  # Change this if needed

    while True:
        ret, frame = screen_capture.read()
        if not ret:
            break

        cv2.imshow('Screen Capture', frame)

        # Take a snapshot every 2 seconds
        timestamp = time.time()
        if timestamp % 2 < 0.1:  # Snapshot condition (approximately every 2 seconds)
            snapshot = frame
            prediction = apply_corrosion_detection(snapshot)
            print(f"Corrosion Detection Result: {prediction}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    screen_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_process()
