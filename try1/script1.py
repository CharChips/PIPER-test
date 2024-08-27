import cv2

def capture_screen():
    # Create a VideoCapture object to capture screen (index 0 for default webcam)
    screen_capture = cv2.VideoCapture(0)  # Change this to capture from screen if needed

    while True:
        ret, frame = screen_capture.read()
        if not ret:
            break

        # Display the captured frame
        cv2.imshow('Screen Capture', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    screen_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_screen()
