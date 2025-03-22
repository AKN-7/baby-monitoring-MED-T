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

# --- Flask Setup ---
app = Flask(__name__)

# --- Model and Camera Setup ---
model_name = "trpakov/vit-face-expression"
try:
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    print("[INFO] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# Function to find available cameras
def find_available_cameras(max_index=5):
    available_cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[INFO] Camera found at index {i}")
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Try to initialize the camera with a fallback
camera_index = 0
camera = cv2.VideoCapture(camera_index)
if not camera.isOpened():
    print(f"[ERROR] Camera at index {camera_index} not accessible")
    available_cameras = find_available_cameras()
    if available_cameras:
        camera_index = available_cameras[0]
        camera = cv2.VideoCapture(camera_index)
        print(f"[INFO] Switched to camera at index {camera_index}")
    else:
        print("[ERROR] No cameras available")
        camera = None

# --- MediaPipe Setup for Facial Landmarks ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Global State for Controls ---
is_streaming = True
mute_notifications = False

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
    # Horizontal distance: left corner to right corner
    C = dist.euclidean(mouth[0], mouth[1])
    # Vertical distances: average of top-left to bottom-left and top-right to bottom-right
    A = dist.euclidean(mouth[2], mouth[4])
    B = dist.euclidean(mouth[3], mouth[5])
    mar = (A + B) / (2.0 * C)
    return mar

def gen_frames():
    global is_streaming, mute_notifications

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("[ERROR] Failed to load Haar Cascade Classifier")
        exit()

    # MediaPipe landmark indices for eyes and mouth
    LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
    MOUTH_POINTS = [61, 291, 405, 321, 375, 287]  # Left, right, top-left, top-right, bottom-left, bottom-right

    # Thresholds for distress detection
    EAR_THRESHOLD_TIGHT = 0.2
    MAR_THRESHOLD_OPEN = 0.7

    while True:
        if not is_streaming:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Stream Paused", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        if camera is None or not camera.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No Camera Feed Available", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        success, frame = camera.read()
        if not success:
            print("[ERROR] Failed to capture frame from camera")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Failed to Capture Frame", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        # Resize frame to ensure consistent dimensions for MediaPipe
        frame = cv2.resize(frame, (640, 480))

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        predicted_label = "None"
        distress_details = []

        # Facial landmark detection with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                try:
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
                except Exception as e:
                    print(f"[ERROR] Error in landmark processing: {e}")
                    distress_details.append("Landmark Error")

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

        is_distressed = False
        distress_message = predicted_label
        if predicted_label.lower() in ["sad", "fear"]:
            is_distressed = True
        if distress_details and ("Tightly Closed" in distress_details[0] or "Wide Open" in distress_details[1]):
            is_distressed = True
            distress_message += " (Distress Indicators Detected)"

        y_offset = 50
        cv2.putText(frame, distress_message, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        for detail in distress_details:
            y_offset += 30
            cv2.putText(frame, detail, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("[ERROR] Failed to encode frame")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"[ERROR] Streaming failed: {e}")
        return Response("Error streaming video", status=500)

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
    if camera is None or not camera.isOpened():
        return {"status": "error", "message": "Camera not accessible"}
    success, frame = camera.read()
    if success:
        snapshot_path = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(snapshot_path, frame)
        return {"status": "success", "message": f"Snapshot saved as {snapshot_path}"}
    return {"status": "error", "message": "Failed to capture snapshot"}

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()