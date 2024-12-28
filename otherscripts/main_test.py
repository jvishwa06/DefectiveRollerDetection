import cv2
import threading
import time
from queue import Queue
from ultralytics import YOLO
import snap7
from snap7.util import set_bool, get_bool

# YOLO model
model = YOLO(r"C:\Users\NBC\Desktop\DefectiveRollerDetection\OldModels\Bigfacelatest.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# PLC connection
plc = snap7.client.Client()
plc.connect("172.17.8.17", 0, 1)

# Shared queue for roller statuses
roller_queue = Queue()

# Frame lock for synchronization
frame_lock = threading.Lock()
current_frame = None

# Add a lock for PLC operations
plc_lock = threading.Lock()

def read_proximity_status(byte_index, bool_index):
    """Read proximity sensor status."""
    with plc_lock:
        data = plc.read_area(snap7.type.Areas.DB, 86, 0, 1)
        return get_bool(data, byte_index=byte_index, bool_index=bool_index)

def trigger_slot_opening(defect_detected):
    """Signal the PLC to open the slot."""
    with plc_lock:
        data = bytearray(2)
        if defect_detected:
            set_bool(data, byte_index=1, bool_index=1, value=True)
        else:
            set_bool(data, byte_index=1, bool_index=0, value=True)
        plc.write_area(snap7.type.Areas.DB, 86, 0, data)

        # Reset signals after a short delay
        #time.sleep(0.2)
        data = bytearray(2)
        set_bool(data, byte_index=1, bool_index=1, value=False)
        set_bool(data, byte_index=1, bool_index=0, value=False)
        plc.write_area(snap7.type.Areas.DB, 86, 0, data)

def capture_frames():
    """Continuously capture frames from the camera."""
    global current_frame
    global cap

    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame

def process_rollers():
    """Process frames and perform YOLO inference."""
    global current_frame
    while True:
        if read_proximity_status(byte_index=0, bool_index=0):  # First proximity sensor
            with frame_lock:
                frame = current_frame.copy()
            defect_class_index = next((key for key, value in model.names.items() if value == 'damage'), None)

            # Perform inference
            results = model.predict(frame, device=0, conf=0.8)
            defect_detected = any(int(result[-1]) == defect_class_index for result in results[0].boxes.data)
            roller_queue.put(defect_detected)

            # Optional: Save frame and annotate for debugging
            for result in results[0].boxes.data:
                (x1, y1, x2, y2) = map(int, result[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(f"roller_{time.time()}.jpg", frame)

def handle_slot_control():
    """Control slot mechanism based on second proximity sensor."""
    while True:
        if read_proximity_status(byte_index=0, bool_index=1):  # Second proximity sensor
            if not roller_queue.empty():
                defect_detected = roller_queue.get()
                trigger_slot_opening(defect_detected)
                print(f"Processed roller: {'Defective' if defect_detected else 'Good'}")

def display_frames():
    """Continuously display frames in a CV2 window."""
    global current_frame
    while True:
        with frame_lock:
            if current_frame is not None:
                cv2.imshow('Real-Time Frame Display', current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            print("Exiting display...")
            break

# Start threads
threads = [
    threading.Thread(target=capture_frames, daemon=True),
    threading.Thread(target=process_rollers, daemon=True),
    threading.Thread(target=handle_slot_control, daemon=True),
    threading.Thread(target=display_frames, daemon=False),  # This thread will block until 'q' is pressed
]

for thread in threads:
    thread.start()

try:
    while True:
        time.sleep(0.1)  # Keep the main thread alive
except KeyboardInterrupt:
    print("Exiting...")

plc.disconnect()
cv2.destroyAllWindows()