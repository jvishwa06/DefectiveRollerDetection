import cv2
import time
from multiprocessing import Process, Array, Queue, Lock
from ultralytics import YOLO
import snap7
from snap7.util import set_bool, get_bool
import numpy as np
import sys

# YOLO model
model = YOLO(r"C:\Users\NBC\Desktop\DefectiveRollerDetection\OldModels\Bigfacelatest.pt")

# Shared frame buffer and roller queue
frame_shape = (960, 1280, 3)
shared_frame = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())
roller_queue = Queue()

# Locks for thread-safe operations
frame_lock = Lock()

def read_proximity_status(plc_client, byte_index, bool_index):
    """Read proximity sensor status."""
    try:
        data = plc_client.read_area(snap7.types.Areas.DB, 86, 0, 1)
        return get_bool(data, byte_index=byte_index, bool_index=bool_index)
    except Exception as e:
        print(f"Error reading proximity status: {e}")
        return False

def trigger_slot_opening(plc_client, defect_detected):
    """Signal the PLC to open the slot."""
    time.sleep(0.09)
    try:
        data = bytearray(2)
        set_bool(data, byte_index=1, bool_index=1, value=defect_detected)
        plc_client.write_area(snap7.types.Areas.DB, 86, 0, data)

        # Reset signals after a short delay
        time.sleep(0.1)
        data = bytearray(2)
        set_bool(data, byte_index=1, bool_index=1, value=False)
        plc_client.write_area(snap7.types.Areas.DB, 86, 0, data)
    except Exception as e:
        print(f"Error triggering slot opening: {e}")

def capture_frames(shared_frame, frame_lock):
    """Continuously capture frames from the camera."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    if not cap.isOpened():
        print("Failed to open camera.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                np_frame = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape)
                np.copyto(np_frame, frame)
        else:
            print("Failed to capture frame.")
            time.sleep(0.1)

    cap.release()

def process_rollers(shared_frame, frame_lock, roller_queue):
    """Process frames for YOLO inference and group defects by rollers."""
    plc = snap7.client.Client()
    try:
        plc.connect("172.17.8.17", 0, 1)
        print("Process Rollers: Connected to PLC.")
    except Exception as e:
        print(f"Process Rollers: PLC connection error: {e}")
        return

    roller_detected = False
    slot_defects = [False, False, False]  # Tracks defects for each slot
    roller_queue_index = [0, 1, 2]  # IDs of rollers in current frame

    while True:
        if read_proximity_status(plc, byte_index=0, bool_index=0) and not roller_detected:
            roller_detected = True

            with frame_lock:
                np_frame = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape)
                frame = np_frame.copy()

            # Perform YOLO inference
            results = model.predict(frame, device=0, conf=0.6)

            # Frame width for slot division
            frame_width = frame.shape[1]
            slot_width = frame_width // 3

            # Process bounding boxes
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box  # Coordinates, confidence, and class
                x_center = (x1 + x2) / 2  # Center of the bounding box
                slot_index = int(x_center // slot_width)  # Determine slot (0, 1, or 2)

                if 0 <= slot_index < 3:
                    slot_defects[slot_index] = slot_defects[slot_index] or True  # OR operation to mark defect

            # Process completed rollers (first in queue)
            roller_status = slot_defects.pop(0)  # First roller's defect status
            roller_queue.put(roller_status)  # Add status to queue
            print(f"Roller ID {roller_queue_index[0]}: {'Defective' if roller_status else 'Good'}")

            # Shift roller IDs and slot_defects for next frame
            slot_defects.append(False)
            roller_queue_index = [roller_queue_index[1], roller_queue_index[2], roller_queue_index[2] + 1]

        elif not read_proximity_status(plc, byte_index=0, bool_index=0):
            roller_detected = False

    plc.disconnect()

def handle_slot_control(roller_queue):
    """Control slot mechanism based on roller defects."""
    plc = snap7.client.Client()
    try:
        plc.connect("172.17.8.17", 0, 1)
        print("Handle Slot Control: Connected to PLC.")
    except Exception as e:
        print(f"Handle Slot Control: PLC connection error: {e}")
        return

    while True:
        if not roller_queue.empty():
            defect_detected = roller_queue.get()
            trigger_slot_opening(plc, defect_detected)
            print(f"Processed roller: {'Defective' if defect_detected else 'Good'}")

    plc.disconnect()

if __name__ == "__main__":
    # Create 'captured' directory if it doesn't exist
    import os
    os.makedirs("captured", exist_ok=True)

    # Define processes
    processes = [
        Process(target=capture_frames, args=(shared_frame, frame_lock), daemon=True),
        Process(target=process_rollers, args=(shared_frame, frame_lock, roller_queue), daemon=True),
        Process(target=handle_slot_control, args=(roller_queue,), daemon=True),
    ]

    # Start processes
    for process in processes:
        process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Main: Exiting...")

    # Terminate processes
    for process in processes:
        process.terminate()
        process.join()



