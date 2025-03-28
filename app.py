import cv2
import torch
import time
import os
from flask import Flask, render_template, Response
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from twilio.rest import Client

app = Flask(__name__)

# --- Model and Camera Setup ---
model_name = "trpakov/vit-face-expression"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
camera = cv2.VideoCapture(0)

# Initialize face detector (using Haar cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Twilio Setup ---
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")
PARENT_PHONE_NUMBER = os.environ.get("PARENT_PHONE_NUMBER")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Alert configuration (in seconds)
DISTRESS_THRESHOLD = 5

# Global state for alerting
distress_start = None
sms_sent = False

def send_sms_alert(emotion, duration):
    message_body = f"Alert: Baby has been distressed for {duration:.1f} seconds. Please check on them."
    try:
        twilio_client.messages.create(
            body=message_body,
            from_=TWILIO_FROM_NUMBER,
            to=PARENT_PHONE_NUMBER
        )
        print("SMS alert sent:", message_body)
    except Exception as e:
        print("Failed to send SMS:", e)

# --- Arducam Pan-Tilt Integration ---
try:
    from arducam_pan_tilt import PanTiltController  # Make sure the module is installed
    pan_tilt = PanTiltController()
except ImportError:
    pan_tilt = None
    print("Arducam pan tilt module not found, pan-tilt functionality will be disabled.")

def gen_frames():
    global distress_start, sms_sent

    while True:
        success, frame = camera.read()
        if not success:
            break

        # --- Face Detection for Tracking ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0 and pan_tilt is not None:
            # Choose the largest detected face (assuming it's the baby)
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            # Draw a rectangle around the detected face (for visualization)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Calculate centers
            face_center_x = x + w / 2
            face_center_y = y + h / 2
            frame_center_x = frame.shape[1] / 2
            frame_center_y = frame.shape[0] / 2

            # Compute offsets
            offset_x = face_center_x - frame_center_x
            offset_y = face_center_y - frame_center_y

            # Thresholds (in pixels) before moving the pan/tilt
            threshold_x = 30
            threshold_y = 30

            # Adjust horizontal position
            if abs(offset_x) > threshold_x:
                if offset_x > 0:
                    pan_tilt.pan_right()  # Baby is to the right; pan right to center
                    print("Pan right")
                else:
                    pan_tilt.pan_left()   # Baby is to the left; pan left to center
                    print("Pan left")

            # Adjust vertical position
            if abs(offset_y) > threshold_y:
                if offset_y > 0:
                    pan_tilt.tilt_down()  # Baby is lower in the frame; tilt down
                    print("Tilt down")
                else:
                    pan_tilt.tilt_up()    # Baby is higher in the frame; tilt up
                    print("Tilt up")

        # --- Emotion Classification for Alerting ---
        # Convert frame to RGB and prepare for the emotion model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        inputs = feature_extractor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        # Alert if distress emotion ("fear" or "sad") is detected
        if predicted_label.lower() in ["fear", "sad"]:
            if distress_start is None:
                distress_start = time.time()
            else:
                elapsed = time.time() - distress_start
                if elapsed >= DISTRESS_THRESHOLD and not sms_sent:
                    send_sms_alert(predicted_label, elapsed)
                    sms_sent = True
        else:
            distress_start = None
            sms_sent = False

        # Overlay the emotion label on the frame
        cv2.putText(
            frame,
            f'Emotion: {predicted_label}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Encode the frame and yield for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
