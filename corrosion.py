import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your pre-trained model (make sure the model is trained for corrosion detection)
model = load_model('saved_model.h5')

# Path to your directory containing images
image_directory = 'images'

def preprocess_image(image):
    # Resize the image to match model's expected input shape
    # image = cv2.resize(image, (224, 224))
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

    # Get list of all images in the directory
    images = [img for img in os.listdir(image_directory) if img.endswith(('.jpg', '.jpeg', '.png'))]

    for image_name in images:
        image_path = os.path.join(image_directory, image_name)
        im = cv2.imread(image_path)

        if im is None:
            print(f"Could not load image {image_name}")
            continue

        # Apply corrosion detection
        predictions = detect_corrosion(im)
        confidence = predictions[0][0]  # Assuming binary classification, adjust index if needed
        label = 'Corrosion Detected' if confidence > 0.5 else 'No Corrosion'
        color = (0, 0, 255) if confidence > 0.5 else (0, 255, 0)
        
        cv2.putText(im, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow('corrosion detection', im)
        key = cv2.waitKey(2000)  # Display each image for 2 seconds (2000 milliseconds)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detection()
