import cv2
import os

# Function to extract the middle frame from a video file
# videopath: Path of the video file
# Returns: The extracted middle frame as an image array
def extract_middle_frame(videopath):
    try:
        # Check if the video file exists
        if not os.path.exists(videopath):
            raise FileNotFoundError(f"Video file not found at {videopath}")

        # Open the video file
        cap = cv2.VideoCapture(videopath)

        # Check if the video was successfully opened
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {videopath}")

        # Get total number of frames and find the middle frame
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_length <= 0:
            raise ValueError(f"Video file appears to be empty: {videopath}")

        middle_frame_number = video_length // 2

        # Set the position to the middle frame and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)
        ret, frame = cap.read()

        # Release the video capture object
        cap.release()

        # Check if the frame was successfully read
        if not ret:
            raise ValueError(f"Failed to read the middle frame from video: {videopath}")

        return frame

    except Exception as e:
        print(f"Error extracting middle frame: {str(e)}")
        return None
