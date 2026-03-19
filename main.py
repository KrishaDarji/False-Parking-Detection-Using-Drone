import cv2
import numpy as np
import subprocess
from ultralytics import YOLO
import os
import pytesseract
import re
import time

# Define the correct local path to your trained YOLOv8 model
model_path = r'best.pt'
number_plate_pattern = re.compile(r'[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4}', re.IGNORECASE)
# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please check the path and ensure the file exists.")
    exit()

# Load the pretrained YOLOv8 model
model = YOLO(model_path)

# FFmpeg path (adjust based on your system)
ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # Update this to your FFmpeg installation path

# Start the drone video stream (example using plutocam)
process = subprocess.Popen(
    ["plutocam", "stream", "start", "--out-file", "-"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Set up FFmpeg to convert the stream to raw video frames
ffmpeg_process = subprocess.Popen(
    [ffmpeg_path, "-i", "-", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
    stdin=process.stdout,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Define drone video resolution (adjust based on your drone)
WIDTH = 2048
HEIGHT = 1152

# Main loop for real-time processing
try:
    while True:
        raw_frame = ffmpeg_process.stdout.read(WIDTH * HEIGHT * 3)
        if not raw_frame or len(raw_frame) != WIDTH * HEIGHT * 3:
            print("Error: End of stream or invalid frame data.")
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
        # Augmentation on that frames like thresholding and sharpening and as my frames may be from night i want to increase the brighteness
        results = model(frame)
        # Inside the while loop, after results = model(frame):
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"Object {class_id}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Drone Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    process.terminate()
    ffmpeg_process.terminate()
    cv2.destroyAllWindows()