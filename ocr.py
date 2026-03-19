import cv2
import numpy as np
from paddleocr import PaddleOCR

def process_video(video_path):
    # Initialize PaddleOCR with your preferred parameters
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_thresh=0.1, det_db_box_thresh=0.1, rec_thresh=0.1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1
        
        # Convert frame from BGR to RGB as PaddleOCR expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run PaddleOCR on the raw frame
        ocr_results = ocr.ocr(rgb_frame, cls=True)

        detected_texts = []
        # If OCR results are returned, annotate the frame
        if ocr_results:
            for line in ocr_results:
                # Each 'line' is a list of detections
                if isinstance(line, list):
                    for detection in line:
                        # Expecting each detection to be in the format: [ [points], [text, confidence] ]
                        if isinstance(detection, list) and len(detection) == 2:
                            box_coords = detection[0]
                            text, confidence = detection[1]
                            detected_texts.append(text)
                            pts = np.array(box_coords, np.int32)
                            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                            # Annotate with detected text and its confidence
                            top_left = (int(box_coords[0][0]), int(box_coords[0][1]))
                            cv2.putText(frame, f"{text} ({confidence:.2f})", (top_left[0], top_left[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Print detected texts for this frame to the command line if any are found
            if detected_texts:
                print(f"Frame {frame_count}: Detected - {' | '.join(detected_texts)}")

        # Display the annotated frame
        cv2.imshow("Video Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = "final_test.mp4"  # Replace with your video file path
process_video(video_path)
