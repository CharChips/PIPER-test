import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your pre-trained model (make sure the model is trained for corrosion detection)
model = load_model('saved_model.h5')

# Path to your video file
video_path = 'pipe-video.mp4'

def preprocess_image_for_model(image):
    # Resize the image to match model's expected input shape
    image = cv2.resize(image, (128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Model expects a batch of images
    image = image / 255.0  # Normalize pixel values
    return image

def detect_corrosion(image):
    preprocessed_image = preprocess_image_for_model(image)
    predictions = model.predict(preprocessed_image)
    return predictions  # Adjust based on your model output (e.g., binary classification or confidence level)

def preprocess_image_for_pipe_joints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    blurred = cv2.GaussianBlur(dilated, (5, 5), 2)
    return blurred

def detect_pipe_joints(image):
    preprocessed_image = preprocess_image_for_pipe_joints(image)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(preprocessed_image, 
                               cv2.HOUGH_GRADIENT, dp=0.5, minDist=20,
                               param1=50, param2=30, minRadius=1, maxRadius=100)
    
    print("Circles Detected:", circles)  # Print circles info
    
    # Draw circles and center dots
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Draw the circle in green
            cv2.circle(image, center, radius, (0, 255, 0), 2)
            # Draw a dot in the center of the circle
            cv2.circle(image, center, 5, (0, 255, 0), -1)  # Green dot with radius 5
    
    return image

def unfold_image(image):
    # Example of a simple polar transformation for unfolding
    # This is a placeholder; actual unfolding might be more complex depending on the pipe's shape and camera angle
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    max_radius = min(center[0], center[1], width - center[0], height - center[1])
    
    polar_image = cv2.linearPolar(image, center, max_radius, cv2.WARP_FILL_OUTLIERS)
    return polar_image

def run_detection():
    cv2.namedWindow("Pipe Joint and Corrosion Detection", cv2.WINDOW_AUTOSIZE)

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

        # Unfold the image
        unfolded_frame = unfold_image(frame)

        # Detect pipe joints
        frame_with_joints = detect_pipe_joints(unfolded_frame)

        # Apply corrosion detection
        predictions = detect_corrosion(frame_with_joints)
        confidence = predictions[0][0]  # Assuming binary classification, adjust index if needed
        label = 'Corrosion Detected' if confidence > 0.5 else 'No Corrosion'
        color = (0, 0, 255) if confidence > 0.5 else (0, 255, 0)
        
        cv2.putText(frame_with_joints, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        cv2.imshow('Pipe Joint and Corrosion Detection', frame_with_joints)
        key = cv2.waitKey(1000)  # Display each frame for 1 second (1000 milliseconds)
        if key == ord('q'):
            break

        # Skip to the next frame after 1 second
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval - 1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_detection()
