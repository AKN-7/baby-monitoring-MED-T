# Baby Distress Detection System

This project uses computer vision and machine learning to monitor a baby's facial expressions in real time. If distress emotions ("fear" or "sad") persist beyond a set threshold, an SMS alert is sent to a parent's phone via Twilio.

## Features

Real-time emotion detection using transformers and OpenCV.

Automatic alert system via Twilio SMS when distress is detected.

Live video stream accessible through a Flask web application.

## Running the Application
- Run the Flask application:
- python app.py
- Then, access the live feed at: http://localhost:5000

## Project Structure
.
├── app.py                 # Main application file
├── templates/
│   ├── index.html         # HTML template for video streaming
├── requirements.txt       # Required Python dependencies
└── README.md              # Project documentation

## How It Works

Captures frames from the webcam.

Converts the frame into a format suitable for ViT (Vision Transformer) model inference.

Uses a pre-trained model (trpakov/vit-face-expression) to classify emotions.

If the detected emotion is "fear" or "sad" and persists for 5+ seconds, an SMS alert is sent to the configured parent’s phone number.

Displays the detected emotion on the video feed.

## Dependencies
- opencv-python
- torch
- transformers
- flask
- twilio
- pillow
