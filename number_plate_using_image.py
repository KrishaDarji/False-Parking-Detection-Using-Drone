import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR

# -------------------------------
# Configuration
# -------------------------------
video_path = 'final_test.mp4'  # Replace with your video file path
csv_path = 'car_owners.csv'  # Replace with your CSV file path
frame_interval = 5  # Process every 5th frame

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load the CSV database into a pandas DataFrame
try:
    df = pd.read_csv(csv_path, dtype=str)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}. Please check the path.")
    exit()
except pd.errors.EmptyDataError:
    print(f"Error: CSV file at {csv_path} is empty.")
    exit()
except pd.errors.ParserError:
    print(f"Error: CSV file at {csv_path} is invalid or corrupted.")
    exit()

# Extract all unique numbers from the DataFrame
database_numbers = set(df.stack().unique())

# -------------------------------
# Video Processing
# -------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}. Please check the path.")
    exit()

frame_count = 0
extracted_numbers = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    if frame_count % frame_interval == 0:
        # Convert frame to RGB (PaddleOCR uses RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform OCR on the frame
        ocr_results = ocr.ocr(rgb_frame, cls=True)

        # Check if OCR results are valid
        if ocr_results is not None:
            for result in ocr_results:
                if result is not None:  # Ensure result is not None
                    for line in result:
                        text = line[1][0]
                        # Check if the detected text is a number
                        if text.isdigit():
                            if text not in extracted_numbers:
                                extracted_numbers.add(text)
                                # Check if the number exists in the CSV database
                                if text in database_numbers:
                                    status = "found in database"
                                else:
                                    status = "not found in database"
                                print(f"Detected Number: {text} - {status}")

                                # Optional: Draw bounding box and text on the frame
                                points = np.array(line[0], dtype=np.int32)  # Convert to NumPy array
                                if points.shape == (4, 2):  # Ensure the shape is (4, 2)
                                    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                                    cv2.putText(frame, f"{text} ({status})", (points[0][0], points[0][1] - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    frame_count += 1

    # Display the frame with annotations
    cv2.imshow('Video Processing', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
