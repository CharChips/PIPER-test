from flask import Flask, Response
import cv2

app = Flask(__name__)

# Replace with the IP address of your ESP32-CAM
ESP32_CAM_URL = "http://192.168.254.36:81/stream"

def gen_frames():
    cap = cv2.VideoCapture(ESP32_CAM_URL)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Use generator to yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
