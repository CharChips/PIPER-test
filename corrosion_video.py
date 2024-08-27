import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your pre-trained model (make sure the model is trained for corrosion detection)
model = load_model('saved_model.h5')

# Path to your video file
video_path = 'data/pipe-video.mp4'

def preprocess_image(image):
    # Resize the image to match model's expected input shape
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Model expects a batch of images
    image = image / 255.0  # Normalize pixel values
    return image

def detect_corrosion(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return predictions  # Adjust based on your model output (e.g., binary classification or confidence level)

def run_detection():
    cv2.namedWindow("corrosion detection", cv2.WINDOW_AUTOSIZE)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    frame_interval = int(frame_rate)  # Snapshot every second

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply corrosion detection
        predictions = detect_corrosion(frame)
        confidence = predictions[0][0]  # Assuming binary classification, adjust index if needed
        label = 'Corrosion Detected' if confidence > 0.5 else 'No Corrosion'
        color = (0, 0, 255) if confidence > 0.5 else (0, 255, 0)
        
        cv2.putText(frame, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow('corrosion detection', frame)
        key = cv2.waitKey(1000)  # Display each frame for 1 second (1000 milliseconds)
        if key == ord('q'):
            break

        # Skip to the next frame after 1 second
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval - 1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detection()
