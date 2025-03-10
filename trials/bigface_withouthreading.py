import cv2
import time
from queue import Queue
from ultralytics import YOLO
import snap7
from snap7.util import set_bool, get_bool

# YOLO model
model = YOLO(r"C:\Users\NBC\Desktop\DefectiveRollerDetection\OldModels\Bigfacelatest.pt")
# PLC connection
plc = snap7.client.Client()
plc.connect("172.17.8.17", 0, 1)

# Queue for roller statuses
roller_queue = Queue()

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Flags for frame processing
roller_detected = False

def read_proximity_status(byte_index, bool_index):
    """Read proximity sensor status."""
    data = plc.read_area(snap7.type.Areas.DB, 86, 0, 1)
    return get_bool(data, byte_index=byte_index, bool_index=bool_index)

def trigger_slot_opening(defect_detected):
    """Signal the PLC to open the slot."""
    data = bytearray(2)
    if defect_detected:
        set_bool(data, byte_index=1, bool_index=1, value=True)
    else:
        set_bool(data, byte_index=1, bool_index=0, value=True)
    plc.write_area(snap7.type.Areas.DB, 86, 0, data)

    # Reset signals after a short delay
    # time.sleep(0.3)  # Increased delay for reliability
    data = bytearray(2)
    set_bool(data, byte_index=1, bool_index=1, value=False)
    set_bool(data, byte_index=1, bool_index=0, value=False)
    plc.write_area(snap7.type.Areas.DB, 86, 0, data)

# Main loop
try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        # Display the frame
        cv2.imshow('Real-Time Frame Display', frame)

        # Check first proximity sensor
        if read_proximity_status(byte_index=0, bool_index=0) and not roller_detected:
            roller_detected = True  # Detect roller
            defect_class_index = next((key for key, value in model.names.items() if value == 'damage'), None)

            # Perform inference on the frame
            results = model.predict(frame, device=0, conf=0.8)
            defect_detected = any(int(result[-1]) == defect_class_index for result in results[0].boxes.data)
            roller_queue.put(defect_detected)

            # Save annotated frame for debugging
            for result in results[0].boxes.data:
                (x1, y1, x2, y2) = map(int, result[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(f"captured//roller_{time.time()}.jpg", frame)

        elif not read_proximity_status(byte_index=0, bool_index=0):
            roller_detected = False  # Reset when roller leaves the sensor

        # Process queue if second proximity sensor is triggered
        if read_proximity_status(byte_index=0, bool_index=1):
            if not roller_queue.empty():
                defect_detected = roller_queue.get()
                trigger_slot_opening(defect_detected)
                print(f"Processed roller: {'Defective' if defect_detected else 'Good'}")

            # Log queue state
            print(f"Queue size: {roller_queue.qsize()}, Contents: {list(roller_queue.queue)}")

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

except KeyboardInterrupt:
    print("Exiting due to KeyboardInterrupt.")

finally:
    # Cleanup
    plc.disconnect()
    cap.release()
    cv2.destroyAllWindows()
