import os
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, send_file, send_from_directory
from ultralytics import YOLO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load trained YOLOv8 model
model = YOLO("best.pt")

# Create necessary folders
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Fix 404 error for favicon.ico
@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"), "favicon.ico", mimetype="image/vnd.microsoft.icon")

# Debug route to check Flask routes
@app.route("/debug_routes")
def debug_routes():
    return str(app.url_map)

# Upload and process video
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return "No file uploaded"
    
    file = request.files["file"]
    if file.filename == "" or not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        return "Invalid file format. Please upload a valid video."

    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Open video
    cap = cv2.VideoCapture(filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video file
    processed_filename = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(processed_filename, fourcc, fps, (width, height))

    frame_count = 0
    frame_skip = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            out.write(frame)
            continue

        # Run YOLO detection
        results = model(frame)

        for result in results:
            if hasattr(result, "boxes") and result.boxes is not None:
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = int(cls)

                    class_names = {0: "Accident", 2: "Car"}
                    class_name = class_names.get(label, "Unknown")

                    color = (0, 255, 0) if label == 2 else (0, 0, 255) if label == 0 else (255, 255, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    return send_file(processed_filename, mimetype="video/mp4")

# 📸 NEW: Upload and process image
@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return "No file uploaded"
    
    file = request.files["file"]
    if file.filename == "" or not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return "Invalid file format. Please upload a valid image."

    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Read image
    img = cv2.imread(filename)
    results = model(img)

    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                label = int(cls)

                class_names = {0: "Accident", 2: "Car"}
                class_name = class_names.get(label, "Unknown")

                color = (0, 255, 0) if label == 2 else (0, 0, 255) if label == 0 else (255, 255, 255)

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save processed image
    processed_filename = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
    cv2.imwrite(processed_filename, img)

    return send_file(processed_filename, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
