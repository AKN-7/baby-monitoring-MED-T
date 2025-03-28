import cv2
import torch
import time
import os
import numpy as np
from flask import Flask, render_template, Response
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import mediapipe as mp
from scipy.spatial import distance as dist
import picamera
import RPi.GPIO as GPIO
from io import BytesIO

# --- Flask Setup ---
app = Flask(__name__)

# --- Model Setup ---
model_name = "trpakov/vit-face-expression"
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    print("[INFO] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# --- Servo Setup for Pan-Tilt ---
GPIO.setmode(GPIO.BCM)
PAN_PIN = 18  # GPIO pin for pan servo
TILT_PIN = 23  # GPIO pin for tilt servo
GPIO.setup(PAN_PIN, GPIO.OUT)
GPIO.setup(TILT_PIN, GPIO.OUT)

pan_servo = GPIO.PWM(PAN_PIN, 50)  # 50Hz PWM
tilt_servo = GPIO.PWM(TILT_PIN, 50)
pan_servo.start(7.5)  # Center position (7.5% duty cycle)
tilt_servo.start(7.5)

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Global State ---
is_streaming = True
mute_notifications = False
pan_angle = 90  # Center position (0-180 degrees)
tilt_angle = 90

# Helper functions for facial feature analysis
def eye_aspect_ratio(eye_points, landmarks, image_shape):
    h, w = image_shape[:2]
    eye = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_points, landmarks, image_shape):
    h, w = image_shape[:2]
    mouth = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in mouth_points]
    C = dist.euclidean(mouth[0], mouth[1])
    A = dist.euclidean(mouth[2], mouth[4])
    B = dist.euclidean(mouth[3], mouth[5])
    mar = (A + B) / (2.0 * C)
    return mar

# Servo control function
def set_servo_angle(servo, angle):
    duty = 2.5 + (angle / 18.0)  # Convert angle (0-180) to duty cycle (2.5-12.5)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.05)  # Allow servo to move
    servo.ChangeDutyCycle(0)  # Stop signal to prevent jitter

def gen_frames():
    global is_streaming, mute_notifications, pan_angle, tilt_angle

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("[ERROR] Failed to load Haar Cascade Classifier")
        exit()

    LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
    MOUTH_POINTS = [61, 291, 405, 321, 375, 287]

    EAR_THRESHOLD_TIGHT = 0.2
    MAR_THRESHOLD_OPEN = 0.7

    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)  # Lower resolution for better performance
        camera.framerate = 24
        stream = BytesIO()

        for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
            if not is_streaming:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(frame, "Stream Paused", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                stream.seek(0)
                stream.truncate()
                continue

            # Read frame from stream
            stream.seek(0)
            data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            stream.seek(0)
            stream.truncate()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

            predicted_label = "None"
            distress_details = []

            # Face tracking with pan-tilt
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2

                # Adjust pan and tilt based on face position
                if face_center_x < frame_center_x - 20:
                    pan_angle = min(180, pan_angle + 2)
                elif face_center_x > frame_center_x + 20:
                    pan_angle = max(0, pan_angle - 2)
                if face_center_y < frame_center_y - 20:
                    tilt_angle = min(180, tilt_angle + 2)
                elif face_center_y > frame_center_y + 20:
                    tilt_angle = max(0, tilt_angle - 2)

                set_servo_angle(pan_servo, pan_angle)
                set_servo_angle(tilt_servo, tilt_angle)

            # Facial landmark detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_ear = eye_aspect_ratio(LEFT_EYE_POINTS, face_landmarks.landmark, frame.shape)
                    right_ear = eye_aspect_ratio(RIGHT_EYE_POINTS, face_landmarks.landmark, frame.shape)
                    avg_ear = (left_ear + right_ear) / 2.0
                    mar = mouth_aspect_ratio(MOUTH_POINTS, face_landmarks.landmark, frame.shape)

                    if avg_ear < EAR_THRESHOLD_TIGHT:
                        distress_details.append("Eyes: Tightly Closed")
                    else:
                        distress_details.append("Eyes: Open/Normal")
                    if mar > MAR_THRESHOLD_OPEN:
                        distress_details.append("Mouth: Wide Open")
                    else:
                        distress_details.append("Mouth: Closed/Normal")

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                inputs = feature_extractor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                predicted_label = model.config.id2label[predicted_class_idx].split("_")[-1]
                break

            is_distressed = predicted_label.lower() in ["sad", "fear"] or \
                           (distress_details and ("Tightly Closed" in distress_details[0] or "Wide Open" in distress_details[1]))
            distress_message = predicted_label + (" (Distress Indicators Detected)" if is_distressed else "")

            y_offset = 30
            cv2.putText(frame, distress_message, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for detail in distress_details:
                y_offset += 20
                cv2.putText(frame, detail, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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

@app.route('/toggle_stream', methods=['POST'])
def toggle_stream():
    global is_streaming
    is_streaming = not is_streaming
    return {"status": "success", "streaming": is_streaming}

@app.route('/mute_notifications', methods=['POST'])
def toggle_notifications():
    global mute_notifications
    mute_notifications = not mute_notifications
    return {"status": "success", "muted": mute_notifications}

@app.route('/capture_snapshot', methods=['POST'])
def capture_snapshot():
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        snapshot_path = f"snapshot_{int(time.time())}.jpg"
        camera.capture(snapshot_path)
        return {"status": "success", "message": f"Snapshot saved as {snapshot_path}"}

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        pan_servo.stop()
        tilt_servo.stop()
        GPIO.cleanup()