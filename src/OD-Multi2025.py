import cv2
import time
from multiprocessing import Process, Array, Queue, Lock
from ultralytics import YOLO
import snap7
from snap7.util import set_bool, get_bool
import numpy as np
import sys

# YOLO model
model = YOLO(r"C:\Users\NBC\Desktop\DefectiveRollerDetection\OldModels\ODlatestmodel.pt")

# Shared frame buffer and roller queue
frame_shape = (960, 1280, 3)
shared_frame = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())
roller_queue = Queue()

# Locks for thread-safe operations
frame_lock = Lock()

def read_proximity_status(plc_client, byte_index, bool_index):
    """Read proximity sensor status."""
    try:
        data = plc_client.read_area(snap7.type.Areas.DB, 86, 0, 1)
       # time.sleep(0.09)
        return get_bool(data, byte_index=byte_index, bool_index=bool_index)
    except Exception as e:
        print(f"Error reading proximity status: {e}")
        return False

def trigger_slot_opening(plc_client, defect_detected):
    """Signal the PLC to open the slot."""
    time.sleep(0.09)
    try:
        data = bytearray(2)
        if defect_detected:
            set_bool(data, byte_index=1, bool_index=3, value=True)
        else:
            set_bool(data, byte_index=1, bool_index=2, value=True)
        plc_client.write_area(snap7.type.Areas.DB, 86, 0, data)

        # Reset signals after a short delay
        #time.sleep(0.1)
        data = bytearray(2)
        set_bool(data, byte_index=1, bool_index=3, value=False)
        set_bool(data, byte_index=1, bool_index=2, value=False)
        plc_client.write_area(snap7.type.Areas.DB, 86, 0, data)
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
            time.sleep(0.1)  # Prevent tight loop on failure

    cap.release()

def process_rollers(shared_frame, frame_lock, roller_queue):
    """Process frames for YOLO inference."""
    # Each process should have its own PLC client
    plc = snap7.client.Client()
    try:
        plc.connect("172.17.8.17", 0, 1)
        print("Process Rollers: Connected to PLC.")
    except Exception as e:
        print(f"Process Rollers: PLC connection error: {e}")
        return  # Exit the process if PLC connection fails

    roller_detected = False

    while True:
        if read_proximity_status(plc, byte_index=1, bool_index=4) and not roller_detected:
            roller_detected = True
            with frame_lock:
                np_frame = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape)
                frame = np_frame.copy()

            # Find the class index for 'damage'
            defect_class_index = next((key for key, value in model.names.items() if value == 'damage'), None)
            if defect_class_index is None:
                print("Defect class 'damage' not found in model.")
                continue

            # Perform inference
            results = model.predict(frame, device=0, conf=0.2)
            defect_detected = False
            for box in results[0].boxes.data:
                if int(box[-1]) == defect_class_index:
                    defect_detected = True
                    break
            roller_queue.put(defect_detected)

            # Save annotated frame for debugging
            for box in results[0].boxes.data:
                (x1, y1, x2, y2) = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            timestamp = int(time.time())
            cv2.imwrite(f"captured/roller_{timestamp}.jpg", frame)

        elif not read_proximity_status(plc, byte_index=0, bool_index=0):
            roller_detected = False

    plc.disconnect()

def handle_slot_control(roller_queue):
    """Control slot mechanism based on second proximity sensor."""
    plc = snap7.client.Client()
    try:
        plc.connect("172.17.8.17", 0, 1)
        print("Handle Slot Control: Connected to PLC.")
    except Exception as e:
        print(f"Handle Slot Control: PLC connection error: {e}")
        return  # Exit the process if PLC connection fails

    while True:
        if read_proximity_status(plc, byte_index=0, bool_index=2):
            if not roller_queue.empty():
                defect_detected = roller_queue.get()
                trigger_slot_opening(plc, defect_detected)
                print(f"Processed roller: {'Defective' if defect_detected else 'Good'}")

            # Log the queue state without accessing the internal list
            queue_size = roller_queue.qsize()
            print(f"Queue size: {queue_size}, Contents: {'Empty' if queue_size == 0 else 'Not Empty'}")


def display_frames(shared_frame, frame_lock):
    """Display frames in a CV2 window."""
    while True:
        with frame_lock:
            np_frame = np.frombuffer(shared_frame.get_obj(), dtype=np.uint8).reshape(frame_shape)
            frame = np_frame.copy()

        cv2.imshow('Real-Time Frame Display', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Display: Exiting...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create 'captured' directory if it doesn't exist
    import os
    os.makedirs("captured", exist_ok=True)

    # Define processes
    processes = [
        Process(target=capture_frames, args=(shared_frame, frame_lock), daemon=True),
        Process(target=process_rollers, args=(shared_frame, frame_lock, roller_queue), daemon=True),
        Process(target=handle_slot_control, args=(roller_queue,), daemon=True),
        Process(target=display_frames, args=(shared_frame, frame_lock), daemon=False)  # Main process
    ]

    # Start processes
    for process in processes:
        process.start()

    try:
        # Keep the main process alive to manage child processes
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Main: Exiting...")

    # Terminate processes
    for process in processes:
        process.terminate()
        process.join()
