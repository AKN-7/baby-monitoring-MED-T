# -*- coding: utf-8 -*-
import torch
import time
import os
import numpy as np
from flask import Flask, render_template, Response
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from scipy.spatial import distance as dist
from io import BytesIO
import subprocess
import io
import platform

# Detect if running on Raspberry Pi
try:
    from PCA9685 import PCA9685
    IS_RASPBERRY_PI = True
except ImportError:
    IS_RASPBERRY_PI = False
    print("[INFO] Not on Raspberry Pi - skipping servo control")

# Check if running on Windows
IS_WINDOWS = platform.system() == "Windows"

# If on Windows, we'll use pygame.camera for webcam access
if IS_WINDOWS:
    try:
        import pygame
        import pygame.camera
        pygame.camera.init()
    except ImportError:
        print("[ERROR] Pygame not installed. Install it with: pip install pygame")
        exit()

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

# --- Servo Setup for Pan-Tilt with PCA9685 (Raspberry Pi only) ---
if IS_RASPBERRY_PI:
    try:
        pwm = PCA9685()
        pwm.setPWMFreq(50)  # Set frequency to 50Hz
    except OSError as e:
        print("Error initializing PCA9685: ", e)
        exit()
else:
    pwm = None  # No servo control on desktop

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Global State ---
is_streaming = True
mute_notifications = False
pan_angle = 90  # Center position (0-180 degrees)
tilt_angle = 90
MAX_ANGLE = 180
MIN_ANGLE = 0

# Initialize webcam for Windows
if IS_WINDOWS:
    try:
        cam_list = pygame.camera.list_cameras()
        if not cam_list:
            print("[ERROR] No webcam found on Windows")
            exit()
        webcam = pygame.camera.Camera(cam_list[0], (320, 240))
        webcam.start()
    except Exception as e:
        print(f"[ERROR] Failed to initialize webcam on Windows: {e}")
        exit()

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

# Servo control function (only active on Raspberry Pi)
def set_servo_angle(channel, angle):
    if IS_RASPBERRY_PI and pwm is not None and MIN_ANGLE <= angle <= MAX_ANGLE:
        pwm.setRotationAngle(channel, angle)
    elif not IS_RASPBERRY_PI:
        print(f"[SIM] Would set channel {channel} to {angle} degrees")

# Function to capture a frame (platform-dependent)
def capture_frame():
    if IS_RASPBERRY_PI:
        # Use libcamera-still on Raspberry Pi
        try:
            process = subprocess.Popen(
                ['libcamera-still', '-o', '-', '--width', '320', '--height', '240', '--nopreview'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            image_data, error = process.communicate()
            if process.returncode != 0:
                print(f"[ERROR] libcamera-still failed: {error.decode()}")
                return None
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            print(f"[ERROR] Failed to capture frame with libcamera: {e}")
            return None
    else:
        # Use pygame.camera on Windows
        try:
            snapshot = webcam.get_image()
            # Convert pygame surface to PIL Image
            pygame_image = pygame.image.tostring(snapshot, "RGB")
            image = Image.frombytes("RGB", (320, 240), pygame_image)
            return image
        except Exception as e:
            print(f"[ERROR] Failed to capture frame with webcam: {e}")
            return None

def gen_frames():
    global is_streaming, mute_notifications, pan_angle, tilt_angle

    LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
    MOUTH_POINTS = [61, 291, 405, 321, 375, 287]

    EAR_THRESHOLD_TIGHT = 0.2
    MAR_THRESHOLD_OPEN = 0.7

    if IS_RASPBERRY_PI:
        set_servo_angle(0, pan_angle)  # Initialize pan
        set_servo_angle(1, tilt_angle)  # Initialize tilt
    else:
        print("[INFO] Servo control skipped - not on Raspberry Pi")

    try:
        while True:
            if not is_streaming:
                # Create a blank image with "Stream Paused" text
                frame = Image.new('RGB', (320, 240), color='black')
                draw = ImageDraw.Draw(frame)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                draw.text((50, 120), "Stream Paused", fill=(255, 255, 255), font=font)
                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG")
                frame_bytes = buffer.getvalue()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue

            # Capture frame (platform-dependent)
            frame = capture_frame()
            if frame is None:
                print("[ERROR] Failed to capture frame")
                continue

            # Convert to RGB for MediaPipe
            frame_rgb = frame.convert('RGB')
            frame_np = np.array(frame_rgb)
            results = face_mesh.process(frame_np)

            predicted_label = "None"
            distress_details = []

            # Face tracking and analysis using MediaPipe
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w = frame.size
                    # Calculate bounding box from landmarks
                    x_coords = [int(lm.x * w) for lm in face_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in face_landmarks.landmark]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    face_center_x = (x_min + x_max) // 2
                    face_center_y = (y_min + y_max) // 2
                    frame_center_x = w // 2
                    frame_center_y = h // 2

                    # Adjust pan and tilt
                    if face_center_x < frame_center_x - 20:
                        pan_angle = min(MAX_ANGLE, pan_angle + 2)
                    elif face_center_x > frame_center_x + 20:
                        pan_angle = max(MIN_ANGLE, pan_angle - 2)
                    if face_center_y < frame_center_y - 20:
                        tilt_angle = min(MAX_ANGLE, tilt_angle + 2)
                    elif face_center_y > frame_center_y + 20:
                        tilt_angle = max(MIN_ANGLE, tilt_angle - 2)

                    set_servo_angle(0, pan_angle)
                    set_servo_angle(1, tilt_angle)

                    # Facial analysis
                    left_ear = eye_aspect_ratio(LEFT_EYE_POINTS, face_landmarks.landmark, (h, w))
                    right_ear = eye_aspect_ratio(RIGHT_EYE_POINTS, face_landmarks.landmark, (h, w))
                    avg_ear = (left_ear + right_ear) / 2.0
                    mar = mouth_aspect_ratio(MOUTH_POINTS, face_landmarks.landmark, (h, w))

                    if avg_ear < EAR_THRESHOLD_TIGHT:
                        distress_details.append("Eyes: Tightly Closed")
                    else:
                        distress_details.append("Eyes: Open/Normal")
                    if mar > MAR_THRESHOLD_OPEN:
                        distress_details.append("Mouth: Wide Open")
                    else:
                        distress_details.append("Mouth: Closed/Normal")

                    # Draw bounding box using Pillow
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(255, 0, 0), width=2)

                    # Emotion detection
                    inputs = feature_extractor(images=frame_rgb, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    predicted_label = model.config.id2label[predicted_class_idx].split("_")[-1]
                    break  # Only process the first face

            # Check for distress
            is_distressed = predicted_label.lower() in ["sad", "fear"] or (distress_details and ("Tightly Closed" in distress_details[0] or "Wide Open" in distress_details[1]))
            distress_message = predicted_label + (" (Distress Indicators Detected)" if is_distressed else "")

            # Draw text on the frame using Pillow
            draw = ImageDraw.Draw(frame)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                small_font = ImageFont.truetype("arial.ttf", 15)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()

            y_offset = 30
            draw.text((10, y_offset), distress_message, fill=(0, 255, 0), font=font)
            for detail in distress_details:
                y_offset += 20
                draw.text((10, y_offset), detail, fill=(0, 255, 255), font=small_font)

            # Encode the frame as JPEG
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG")
            frame_bytes = buffer.getvalue()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        if IS_WINDOWS and 'webcam' in globals():
            webcam.stop()

# Flask routes
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
    if IS_RASPBERRY_PI:
        snapshot_path = f"snapshot_{int(time.time())}.jpg"
        subprocess.run(
            ['libcamera-still', '-o', snapshot_path, '--width', '320', '--height', '240', '--nopreview']
        )
        return {"status": "success", "message": f"Snapshot saved as {snapshot_path}"}
    else:
        # On Windows, save a snapshot using pygame
        frame = capture_frame()
        if frame is None:
            return {"status": "error", "message": "Failed to capture snapshot"}
        snapshot_path = f"snapshot_{int(time.time())}.jpg"
        frame.save(snapshot_path, "JPEG")
        return {"status": "success", "message": f"Snapshot saved as {snapshot_path}"}

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        if IS_WINDOWS and 'webcam' in globals():
            webcam.stop()