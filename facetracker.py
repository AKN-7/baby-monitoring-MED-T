import cv2
import os
import numpy as np
import time
from pantilthat import pan, tilt, set_pixel_rgbw, show, light_mode, WS2812

# Load the V4L2 module for the Pi Camera (if needed)
os.system('sudo modprobe bcm2835-v4l2')

def face_tracker(camera_stream):
    cascade_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("Error loading cascade classifier from:", cascade_path)
        return

    # Initial pan/tilt settings
    cam_pan = 90
    cam_tilt = 45

    # Set the default position and initialize light mode
    pan(cam_pan - 90)
    tilt(cam_tilt - 90)
    light_mode(WS2812)

    def lights(r, g, b, w):
        for x in range(18):
            # Light up specific pixels (customize as needed)
            set_pixel_rgbw(x, r if x in [3,4] else 0, g if x in [3,4] else 0, b, w if x in [0,1,6,7] else 0)
        show()

    lights(0, 0, 0, 50)

    while True:
        ret, frame = camera_stream.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(15, 15))
        if len(faces) > 0:
            lights(0, 50, 0, 50)  # Green light when a face is detected
            (x, y, w, h) = faces[0]
            midFaceX = x + w // 2
            midFaceY = y + h // 2

            # Calculate offsets relative to the center of the frame
            offsetX = (midFaceX / (frame.shape[1] / 2)) - 1
            offsetY = (midFaceY / (frame.shape[0] / 2)) - 1

            cam_pan -= offsetX * 5
            cam_tilt += offsetY * 5

            # Clamp values between 0 and 180
            cam_pan = max(0, min(180, cam_pan))
            cam_tilt = max(0, min(180, cam_tilt))

            print(f"Face tracker offsets: ({offsetX:.2f}, {offsetY:.2f}), pan: {cam_pan:.0f}, tilt: {cam_tilt:.0f}")

            pan(int(cam_pan - 90))
            tilt(int(cam_tilt - 90))
        else:
            lights(50, 0, 0, 50)  # Red light when no face is detected

        # Slow down the loop slightly to reduce CPU usage
        time.sleep(0.05)
