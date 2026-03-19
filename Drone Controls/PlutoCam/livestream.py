import cv2
import numpy as np
import subprocess
import plutocam

# Path to your ffmpeg executable
ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'

def start_video_stream():
    """
    Start the drone stream using plutocam and pipe it through ffmpeg to output raw frames.
    """
    # Start the drone video stream using plutocam
    process = subprocess.Popen(
        ["plutocam", "stream", "start", "--out-file", "-"],
        stdout=subprocess.PIPE
    )
    
    # Use ffmpeg to convert the stream to raw video frames in BGR24 format
    ffmpeg_process = subprocess.Popen(
        [ffmpeg_path, "-i", "-", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdin=process.stdout,
        stdout=subprocess.PIPE
    )
    return process, ffmpeg_process

def main():
    # Initialize the drone and set the time (if required by your drone API)
    drone = plutocam.LWDrone()
    drone.set_time()
    
    # Start the streaming processes
    process, ffmpeg_process = start_video_stream()

    # Set the expected resolution of the video stream.
    # Adjust these values to match your drone camera's output.
    width = 2048
    height = 1152
    frame_size = width * height * 3  # 3 bytes per pixel for BGR24

    while True:
        # Read a full frame's worth of data
        raw_frame = ffmpeg_process.stdout.read(frame_size)
        
        # If we did not get the full frame, try reinitializing the stream
        if len(raw_frame) != frame_size:
            print("Incomplete frame received. Reinitializing the stream...")
            process.terminate()
            ffmpeg_process.terminate()
            process, ffmpeg_process = start_video_stream()
            continue
        
        try:
            # Convert the raw frame bytes into a NumPy array and reshape it to an image
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
        except Exception as e:
            print("Error processing frame:", e)
            continue

        # Display the frame using OpenCV
        cv2.imshow('Drone Video Stream', frame)
        
        # Check for user input to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Clean up: terminate the subprocesses and close the display window
    process.terminate()
    ffmpeg_process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
