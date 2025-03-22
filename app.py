import cv2
import torch
import time
import os
from flask import Flask, render_template, Response
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from twilio.rest import Client
from pantilthat import *

app = Flask(__name__)

# --- Model and Camera Setup ---
model_name = "trpakov/vit-face-expression"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
camera = cv2.VideoCapture(0)

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

# Camera Movement Settings
cam_pan = 90
cam_tilt = 45
pan(cam_pan - 90)
tilt(cam_tilt - 90)

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

def gen_frames():
    global distress_start, sms_sent, cam_pan, cam_tilt

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        midFace = None

        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Get face center
            midFaceX = x + (w // 2)
            midFaceY = y + (h // 2)
            midFace = (midFaceX, midFaceY)

            # Calculate offsets
            offsetX = (midFaceX / (frame.shape[1] / 2)) - 1
            offsetY = (midFaceY / (frame.shape[0] / 2)) - 1

            # Adjust camera position
            cam_pan -= int(offsetX * 5)
            cam_tilt += int(offsetY * 5)
            cam_pan = max(0, min(180, cam_pan))
            cam_tilt = max(0, min(180, cam_tilt))

            pan(cam_pan - 90)
            tilt(cam_tilt - 90)

            break  # Track only the first detected face

        # Convert frame to RGB and prepare for model processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Preprocess and infer emotion
        inputs = feature_extractor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        # Check for distress emotions ("fear" or "sad") and alert if threshold exceeded
        if predicted_label.lower() in ["fear", "sad"]:
            if distress_start is None:
                distress_start = time.time()
            else:
                elapsed = time.time() - distress_start
                if elapsed >= DISTRESS_THRESHOLD and not sms_sent:
                    send_sms_alert(predicted_label, elapsed)
                    sms_sent = True
        else:
            # Reset when a non-distress emotion is detected
            distress_start = None
            sms_sent = False

        # Overlay the emotion on the frame
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

        # Encode as JPEG and yield for streaming
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
