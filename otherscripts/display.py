import streamlit as st
from ultralytics import YOLO
import cv2
import threading
import random
import time

# Simulated proximity sensor
class ProximitySensorProxy:
    """Simulates a proximity sensor."""
    def __init__(self):
        self.status = False  # Default proximity status

    def read_status(self) -> bool:
        """Simulates reading the proximity sensor status."""
        return self.status

    def toggle_status(self):
        """Randomly toggles the proximity sensor status."""
        self.status = random.choice([True, False])

# Create a proxy object
proximity_sensor = ProximitySensorProxy()

# Load YOLO model
model = YOLO(r"OldModels\Bigfacelatest.pt")

# Start camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FPS, 30)

# Shared state
shared_state = {
    "proximity_status": False,
    "last_frame": None,
    "last_defect_status": False,
}

# Function to process inference
def process_inference(frame):
    """Run YOLO inference."""
    results = model.predict(frame, device=0, conf=0.5)
    defect_detected = False

    for result in results[0].boxes.data:
        class_id = int(result[-1])
        class_name = model.names[class_id]
        if class_name == "damage":
            defect_detected = True
            (x1, y1, x2, y2) = map(int, result[:4])  # Bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                class_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            break

    return defect_detected, frame

# Background task to simulate proximity sensor
def simulate_proximity_sensor(shared_state):
    """Simulates the proximity sensor status."""
    while True:
        shared_state["proximity_status"] = random.choice([True, False])
        time.sleep(1)

# Background task to read and process frames
def read_and_process_frames(shared_state):
    """Capture frames and process them."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Process inference
        defect_detected, processed_frame = process_inference(frame)

        # Update shared state
        shared_state["last_frame"] = processed_frame
        shared_state["last_defect_status"] = defect_detected

        time.sleep(0.1)

# Initialize Streamlit
st.title("YOLO Object Detection and Simulated Proximity Sensor")

# Create an empty container for the image
image_placeholder = st.empty()

if st.button("Start Monitoring"):
    # Start background threads
    threading.Thread(target=simulate_proximity_sensor, args=(shared_state,), daemon=True).start()
    threading.Thread(target=read_and_process_frames, args=(shared_state,), daemon=True).start()

# Display results in Streamlit
if shared_state["last_frame"] is not None:
    # Convert the frame to RGB format before displaying
    frame_rgb = cv2.cvtColor(shared_state["last_frame"], cv2.COLOR_BGR2RGB)
    
    # Update the displayed image in real-time
    image_placeholder.image(frame_rgb, caption="Processed Frame", use_column_width=True)

# Display other information
st.write(f"Defect Detected: {shared_state['last_defect_status']}")
st.write(f"Proximity Sensor Status: {shared_state['proximity_status']}")

# Shutdown resources
if st.button("Shutdown"):
    cap.release()
    cv2.destroyAllWindows()
    st.write("Resources released. Application shutting down.")
