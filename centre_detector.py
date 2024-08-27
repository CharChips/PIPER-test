import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=2)
    blurred = cv2.GaussianBlur(dilated, (5, 5), 2)
    return blurred

def detect_pipe_joints(image):
    # Create a color version of the image to draw detections
    color_image = image.copy()

    preprocessed_image = preprocess_image(image)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(preprocessed_image, 
                               cv2.HOUGH_GRADIENT, dp=0.5, minDist=20,
                               param1=50, param2=30, minRadius=1, maxRadius=100)
    
    print("Circles Detected:", circles)  # Print circles info
    
    # Draw circles and center dots on the color image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Draw the circle in green on the color image
            cv2.circle(color_image, center, radius, (0, 255, 0), 2)
            # Draw a dot in the center of the circle in green
            cv2.circle(color_image, center, 5, (0, 255, 0), -1)  # Green dot with radius 5
    
    return color_image

if __name__ == '__main__':
    image_path = 'data/inv.png'
    image = cv2.imread(image_path)
    
    image_with_detections = detect_pipe_joints(image)
    
    # Show the color image with detections
    cv2.imshow('Pipe Joint Detection', image_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the color image with detections
    cv2.imwrite('output_with_detections.jpg', image_with_detections)
