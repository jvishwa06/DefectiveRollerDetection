import snap7
import time
import cv2
from snap7.util import get_bool
import os

# Initialize and connect to the PLC
plc = snap7.client.Client()
plc.connect("172.17.8.17", 0, 1)

# Initialize the camera (use index 0 to refer to the default camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Folder to save captured frames
capture_folder = "captured_frames"
if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)

def read_proximity_status(byte_index, bool_index):
    """Read proximity sensor status."""
    data = plc.read_area(snap7.type.Areas.DB, 86, 0, 1)
    return get_bool(data, byte_index=byte_index, bool_index=bool_index)

def capture_frame(frame):
    """Capture the flipped frame when proximity is detected."""
    # Flip the frame (flip vertically, use 1 for horizontal, 0 for vertical, or -1 for both)
    flipped_frame = cv2.flip(frame, 0)  # Flip vertically (0 for vertical flip)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(capture_folder, f"captured_frame_{timestamp}.jpg")
    cv2.imwrite(filename, flipped_frame)
    print(f"Flipped Frame captured: {filename}")

try:
    while True:
        start_time = time.time()

        # Read proximity status
        proximity_status = read_proximity_status(0, 2)

        # Capture frame if proximity detected
        ret, frame = cap.read()
        if ret:
            flipped_frame = cv2.flip(frame, 0)
            # Display the live feed without flipping
            cv2.imshow("Live Feed", flipped_frame)

            # Capture and save the flipped frame if proximity detected
            if proximity_status:
                capture_frame(frame)

        # Record the end time
        end_time = time.time()

        # Calculate and print the time taken
        time_taken = end_time - start_time
        print(f"Proximity Status: {proximity_status}, Time Taken: {time_taken:.6f} seconds")

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Disconnect from the PLC and release the camera when done
    plc.disconnect()
    cap.release()
    cv2.destroyAllWindows()
