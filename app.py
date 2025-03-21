import cv2
import torch
import time
import os
from flask import Flask, render_template, Response
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from twilio.rest import Client
import threading

# Import the shared camera stream and face tracker modules
from camera_stream import CameraStream
from facetracker import face_tracker

app = Flask(__name__)

# --- Shared Camera Setup ---
camera_stream = CameraStream(src=0).start()

# --- Model Setup for Emotion Recognition ---
model_name = "trpakov/vit-face-expression"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

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
    message_body = (
        f"Alert: Baby has been {emotion.lower()} for {duration:.1f} seconds. Please check on them."
    )
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
    global distress_start, sms_sent

    while True:
        ret, frame = camera_stream.read()
        if not ret or frame is None:
            continue

        # Convert frame to RGB and then to PIL Image for model processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Preprocess and run emotion recognition
        inputs = feature_extractor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        # Check for distress emotions and trigger alert if needed
        if predicted_label.lower() in ["fear", "sad", "happy", "surprised", "angry"]:
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

        # Encode frame as JPEG for streaming
        ret2, buffer = cv2.imencode('.jpg', frame)
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
    # Start the face tracker in a background thread, using the shared camera stream
    face_tracker_thread = threading.Thread(target=face_tracker, args=(camera_stream,))
    face_tracker_thread.daemon = True
    face_tracker_thread.start()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
