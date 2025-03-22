import cv2
import torch
import time  # Needed for Twilio distress timing (commented out for now)
import os    # Needed for Twilio environment variables (commented out for now)
from flask import Flask, render_template, Response
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
# from twilio.rest import Client  # Commented out for desktop testing

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

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("[ERROR] Camera not accessible")
    exit()

# --- Twilio Setup ---
# TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
# TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
# TWILIO_FROM_NUMBER = os.environ.get("TWILIO_FROM_NUMBER")
# PARENT_PHONE_NUMBER = os.environ.get("PARENT_PHONE_NUMBER")
# twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Alert configuration (in seconds)
# DISTRESS_THRESHOLD = 5

# Global state for alerting and controls
# distress_start = None
# sms_sent = False
is_streaming = True  # For pause/resume functionality
mute_notifications = False  # For mute notifications toggle

# Camera Movement Settings
# cam_pan = 90
# cam_tilt = 45
# pan(cam_pan - 90)
# tilt(cam_tilt - 90)

# def send_sms_alert(emotion, duration):
#     message_body = f"Alert: Baby has been distressed for {duration:.1f} seconds. Please check on them."
#     try:
#         twilio_client.messages.create(
#             body=message_body,
#             from_=TWILIO_FROM_NUMBER,
#             to=PARENT_PHONE_NUMBER
#         )
#         print("SMS alert sent:", message_body)
#     except Exception as e:
#         print("Failed to send SMS:", e)

def gen_frames():
    # global distress_start, sms_sent, cam_pan, cam_tilt  # Commented out for desktop testing
    global is_streaming, mute_notifications

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("[ERROR] Failed to load Haar Cascade Classifier")
        exit()

    snapshot_counter = 0  # For naming snapshot files

    while True:
        if not is_streaming:
            # If streaming is paused, yield an empty frame to keep the connection alive
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            continue

        success, frame = camera.read()
        if not success:
            print("[ERROR] Failed to capture frame from camera")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        predicted_label = "None"  # Default if no face is detected

        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Emotion detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            inputs = feature_extractor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx].split("_")[-1]  # Extract single word (e.g., "happy")
            break  # Process only the first face

        # # Check for distress emotions ("fear" or "sad") and alert if threshold exceeded (commented out)
        # if predicted_label.lower() in ["fear", "sad"]:
        #     if distress_start is None:
        #         distress_start = time.time()
        #     else:
        #         elapsed = time.time() - distress_start
        #         if elapsed >= DISTRESS_THRESHOLD and not sms_sent:
        #             send_sms_alert(predicted_label, elapsed)
        #             sms_sent = True
        # else:
        #     # Reset when a non-distress emotion is detected
        #     distress_start = None
        #     sms_sent = False

        # Overlay just the emotion word on the frame
        cv2.putText(
            frame,
            predicted_label,
            (10, 50),  # Position near top-left
            cv2.FONT_HERSHEY_SIMPLEX,
            2,  # Larger font size for visibility
            (0, 255, 0),  # Green text
            3,  # Thickness
            cv2.LINE_AA
        )

        # Encode as JPEG and yield for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("[ERROR] Failed to encode frame")
            break
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
    success, frame = camera.read()
    if success:
        snapshot_path = f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(snapshot_path, frame)
        return {"status": "success", "message": f"Snapshot saved as {snapshot_path}"}
    return {"status": "error", "message": "Failed to capture snapshot"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)