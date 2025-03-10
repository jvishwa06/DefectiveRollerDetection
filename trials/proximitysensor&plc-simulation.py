import streamlit as st
from ultralytics import YOLO
import cv2
import threading
import asyncio
import random
import torch

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# Load YOLO model on CUDA or CPU depending on availability
model = YOLO("OldModels\\Bigfacelatest.pt")  # Make sure your path is correct
model.to(device)  # Move the model to GPU if available

# Start camera using DirectShow on Windows (this opens the camera more quickly)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_FPS, 30)

# Shared state for communication between threads and async tasks
shared_state = {
    "proximity_status": False,
    "last_frame": None,
    "last_defect_status": False,
}

# Function to process inference (inference running on GPU or CPU)
async def process_inference(frame):
    """Run YOLO inference on the frame using CUDA or CPU."""
    results = model.predict(frame, device=device, conf=0.5)
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
async def simulate_proximity_sensor(shared_state):
    """Simulates the proximity sensor status."""
    while True:
        shared_state["proximity_status"] = random.choice([True, False])
        await asyncio.sleep(1)  # Use asyncio.sleep for non-blocking delay

# Background task to read and process frames (Threaded)
def read_and_process_frames(shared_state, loop):
    """Capture frames from the camera and process them."""
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")  # Debugging message
            continue

        # Run async inference inside the existing event loop
        defect_detected, processed_frame = loop.run_until_complete(process_inference(frame))

        # Update shared state
        shared_state["last_frame"] = processed_frame
        shared_state["last_defect_status"] = defect_detected

        # Add a small delay to allow other tasks to run
        time.sleep(0.1)  # Note: This is blocking, but doesn't block async tasks

# Initialize Streamlit
st.title("YOLO Object Detection and Simulated Proximity Sensor")

# Create an empty container for the image
image_placeholder = st.empty()

# Define the function to start monitoring
async def start_monitoring():
    """Start background tasks with asyncio."""
    # Start the proximity sensor simulation
    await asyncio.gather(
        simulate_proximity_sensor(shared_state)
    )

# Function to start threading and asyncio together
def start_background_tasks():
    """Start both asyncio and threading tasks concurrently."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)  # Set the event loop for the current thread

    # Start the proximity sensor simulation in an asyncio loop
    asyncio.run(start_monitoring())

    # Start the camera frame processing in a separate thread
    threading.Thread(target=read_and_process_frames, args=(shared_state, loop), daemon=True).start()

# Display results in Streamlit
if st.button("Start Monitoring"):
    start_background_tasks()

# Display the most recent frame
if shared_state["last_frame"] is not None:
    # Convert the frame to RGB format before displaying
    frame_rgb = cv2.cvtColor(shared_state["last_frame"], cv2.COLOR_BGR2RGB)
    
    # Update the displayed image in real-time
    image_placeholder.image(frame_rgb, caption="Processed Frame", use_column_width=True)

st.write(f"Defect Detected: {shared_state['last_defect_status']}")
st.write(f"Proximity Sensor Status: {shared_state['proximity_status']}")

# Shutdown resources
if st.button("Shutdown"):
    cap.release()
    cv2.destroyAllWindows()
    st.write("Resources released. Application shutting down.")
