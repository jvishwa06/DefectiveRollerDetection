import cv2
import time
from multiprocessing import Process, Array, Queue, Lock, Value, Manager
from ultralytics import YOLO
import snap7
from snap7.util import set_bool, get_bool
import numpy as np
import sys
from snap7.type import Areas
import csv
import os

def plc_communication(plc_ip, rack, slot, db_number, shared_data, command_queue):
    """
    Handles all PLC communication: reading sensor statuses and executing commands.
    """
    plc_client = snap7.client.Client()
    try:
        plc_client.connect(plc_ip, rack, slot)
        print("PLC Communication: Connected to PLC.")
    except Exception as e:
        print(f"PLC Communication: Connection error: {e} ⚠️")
        return

    try:
        while True:
            try:
                data = plc_client.read_area(Areas.DB, db_number, 0, 2)
                shared_data['bigface_presence'] = get_bool(data, byte_index=0, bool_index=0)
                shared_data['od_presence'] = get_bool(data, byte_index=1, bool_index=4)
                shared_data['bigface'] = get_bool(data, byte_index=0, bool_index=1)
                shared_data['od'] = get_bool(data, byte_index=0, bool_index=2)
            except Exception as e:
                print(f"PLC Communication: Error reading sensors: {e} ⚠️")
                

            while not command_queue.empty():
                try:
                    command, params = command_queue.get_nowait()
                    if command == 'accept_bigface':
                        trigger_plc_action(plc_client, db_number, byte_index=1, bool_index=0, action="accept")
                    elif command == 'reject_bigface':
                        trigger_plc_action(plc_client, db_number, byte_index=1, bool_index=1, action="reject")
                    elif command == 'accept_od':
                        trigger_plc_action(plc_client, db_number, byte_index=1, bool_index=2, action="accept")
                    elif command == 'reject_od':
                        trigger_plc_action(plc_client, db_number, byte_index=1, bool_index=3, action="reject")
                    else:
                        print(f"PLC Communication: Unknown command: {command}")
                except Exception as e:
                    print(f"PLC Communication: Error handling command: {e} ⚠️")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("PLC Communication: KeyboardInterrupt received. Disconnecting PLC.")
    finally:
        plc_client.disconnect()
        print("PLC Communication: Disconnected from PLC.")

def trigger_plc_action(plc_client, db_number, byte_index, bool_index, action):
    """Signal the PLC to perform an action (accept/reject)."""
    try:
        print(f"PLC Action: Triggering {action.upper()} slot at byte {byte_index}, bit {bool_index}...")
        data = bytearray(2)
        set_bool(data, byte_index=byte_index, bool_index=bool_index, value=True)
        plc_client.write_area(Areas.DB, db_number, 0, data)

        time.sleep(0.1)
        set_bool(data, byte_index=byte_index, bool_index=bool_index, value=False)
        plc_client.write_area(Areas.DB, db_number, 0, data)
        #print(f"PLC Action: {action.upper()} slot reset.")
    except Exception as e:
        print(f"PLC Action: Error triggering {action.upper()} slot: {e} ⚠️")

def capture_frames_bigface(shared_frame_bigface, frame_lock_bigface,frame_shape):
    """Continuously capture frames from the camera."""
    print("Starting frame capture...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    if not cap.isOpened():
        print("Failed to open camera.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock_bigface:
                np_frame = np.frombuffer(shared_frame_bigface.get_obj(), dtype=np.uint8).reshape(frame_shape)
                np.copyto(np_frame, frame)
        else:
            print("Failed to capture frame.")
            time.sleep(0.1)

    cap.release()

# The rest of the Bigface and OD logic remains untouched.

def handle_slot_control_bigface(roller_queue_bigface,shared_data,command_queue):
    """Control slot mechanism based on second proximity sensor."""
    global roller_number
    print("Starting slot control...")

    a = False
    while True:
        if shared_data["bigface"] and not a:
            a = True
            if not roller_queue_bigface.empty():
                defect_detected = roller_queue_bigface.get()
                status = "Defective" if defect_detected else "Good"
                print(f"Slot control received roller status: {status}")
                command_queue.put(("accept_bigface" if not defect_detected else "reject_bigface", None))
        elif not shared_data["bigface"]:
            a = False

def process_rollers_bigface(shared_frame_bigface, frame_lock_bigface, roller_queue_bigface,model_bigface,proximity_count_bigface,roller_updation_dict,queue_lock,shared_data,frame_shape):
    """Process frames for YOLO inference."""
    print("Starting roller processing...")
    black_frame = np.zeros(frame_shape, dtype=np.uint8)
    # print("Processing black image with YOLO before starting main loop...")

    # Step 2: Perform YOLO Inference on the Black Image
    try:
        results = model_bigface.predict(black_frame, device=0, conf=0.3)
        print("Black image YOLO processing complete.")
        # Optionally, handle the results if needed
    except Exception as e:
        print(f"Error during YOLO inference on black image: {e}")


    roller_detected = False
    
    while True:
        if shared_data["bigface_presence"] and not roller_detected:
            roller_detected = True
            print("Roller detected. Capturing frame...")
            with frame_lock_bigface:
                np_frame = np.frombuffer(shared_frame_bigface.get_obj(), dtype=np.uint8).reshape(frame_shape)
                frame = np_frame.copy()
            pc = 0
        
            proximity_count_bigface.value += 1
            pc = proximity_count_bigface.value

            defect_class_index = next((key for key, value in model_bigface.names.items() if value == 'damage'), None)
            if defect_class_index is None:
                print("Defect class 'damage' not found in model.")
                continue

            # Perform inference
            results = model_bigface.predict(frame, device=0, conf=0.5)
            # print(f"Inference results: {results}")
            defect_detected = any(int(box[-1]) == defect_class_index for box in results[0].boxes.data)
            roller_queue_bigface.put(defect_detected)
            with queue_lock:
                if(defect_detected):
                    roller_updation_dict[pc]=1
                else:
                    roller_updation_dict[pc]=0
            print("roller dict",roller_updation_dict)
            # Debugging the queue
            queue_list = []
            while not roller_queue_bigface.empty():
                item = roller_queue_bigface.get()
                queue_list.append(item)
            for item in queue_list:
                roller_queue_bigface.put(item)
            print(f"Queue after adding element: {queue_list}")
            
            # Save annotated frame for debugging
            for box in results[0].boxes.data:
                (x1, y1, x2, y2) = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            timestamp = int(time.time())
            cv2.imwrite(f"captured/roller_{timestamp}.jpg", frame)
            print("Prox",pc)

        elif not shared_data['bigface_presence']:
            roller_detected = False


def capture_frames_od(shared_frame_od, frame_lock_od,frame_shape):
    """Continuously captureframes from the camera."""
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[0])

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, -1)
            with frame_lock_od:
                np_frame = np.frombuffer(shared_frame_od.get_obj(), dtype=np.uint8).reshape(frame_shape)
                np.copyto(np_frame, frame)
        else:
            print("Failed to capture frame.")
            time.sleep(0.01)


def process_frames_od(shared_frame_od, frame_lock_od, roller_data_od, proximity_count_od, roller_queue_od, queue_lock, roller_updation_dict,shared_data,frame_shape):
    """Process frames for YOLO inference."""
    detected_folder = "detected_frames"
    os.makedirs(detected_folder, exist_ok=True)
    black_frame = np.zeros(frame_shape, dtype=np.uint8)
    model_path = r"OldModels\ODlatestmodel.pt"
    yolo = YOLO(model_path)
    # print("Processing black image with YOLO before starting main loop...")

    # Step 2: Perform YOLO Inference on the Black Image
    try:
        results = yolo.predict(black_frame, device=0, conf=0.3)
        print("Black image YOLO processing complete.")
        # Optionally, handle the results if needed
    except Exception as e:
        print(f"Error during YOLO inference on black image: {e}")


    roller_detected = False

    while True:
        if shared_data["od_presence"] and not roller_detected:
            roller_detected = True
            # Process frame and YOLO logic
            with frame_lock_od:
                np_frame = np.frombuffer(shared_frame_od.get_obj(), dtype=np.uint8).reshape(frame_shape)
                frame = np_frame.copy()
            pc = 0
            queue_list = []
            proximity_count_od.value += 1
            pc = proximity_count_od.value
            # Run YOLO inference
            
            results = yolo.predict(frame, device=0, conf=0.2,save=True)
            detections = [
                ("roller" if int(box[-1]) == 4 else "defect", int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                for box in results[0].boxes.data
            ] if results and results[0].boxes.data is not None else []

            # Separate rollers and defects
            roller_boxes = []
            defect_boxes = []

            for bbox in detections:
                class_name, x_min, y_min, x_max, y_max = bbox
                if class_name == "roller":
                    roller_boxes.append((x_min, y_min, x_max, y_max))
                else:
                    defect_boxes.append((class_name, x_min, y_min, x_max, y_max))

            # Sort rollers left to right by x_min
            roller_boxes.sort(key=lambda box: box[0])
            if pc <= 3:
                # Update roller defect status in the shared dictionary
                for i, roller in enumerate(roller_boxes, start=1):
                    roller_id = f"roller_{i}"
                    has_defect = False
                    for defect in defect_boxes:
                        _, x_min, y_min, x_max, y_max = defect
                        # Check intersection
                        if not (roller[2] < x_min or x_max < roller[0] or roller[3] < y_min or y_max < roller[1]):
                            has_defect = True
                            break

                    # Use OR operation to retain defect status across frames
                    roller_data_od[roller_id] = roller_data_od.get(roller_id, False) or has_defect
            else:
                for i, roller in enumerate(roller_boxes, start=(pc - 2)):
                    roller_id = f"roller_{i}"
                    has_defect = False
                    for defect in defect_boxes:
                        _, x_min, y_min, x_max, y_max = defect
                        # Check intersection
                        if not (roller[2] < x_min or x_max < roller[0] or roller[3] < y_min or y_max < roller[1]):
                            has_defect = True
                            break

                    # Use OR operation to retain defect status across frames
                    roller_data_od[roller_id] = roller_data_od.get(roller_id, False) or has_defect
            if(pc>=3):
                roller_id = f"roller_{pc-2}"
                with queue_lock:
                    if roller_updation_dict[pc-2]==0:
                        defect1=roller_data_od.get(roller_id)
                        print("has defect:",defect1)
                        roller_queue_od.put(defect1)
            while not roller_queue_od.empty():
                item = roller_queue_od.get()
                queue_list.append(item)
            for item in queue_list:
                roller_queue_od.put(item)
            print(f"Queue after adding element od: {queue_list}")
            

            # Display for debugging
            print("Shared Roller Data od:", dict(roller_data_od))
            print("Proximity Count:", proximity_count_od.value)
            
        elif not shared_data["od_presence"]:
            roller_detected = False
def handle_slot_control_od(roller_queue_od,roller_data_od,shared_data,command_queue):
    """Control slot mechanism based on second proximity sensor."""

    a=False
    while True:
        if shared_data["od"] and not a:
            a=True
            if not roller_queue_od.empty():
                defect_detected = roller_queue_od.get()
                print("Trigger:",defect_detected)
                command_queue.put(("accept_od" if not defect_detected else "reject_od", None))
                print(f"Processed roller: {'Defective' if defect_detected else 'Good'}")

            # Log the queue state
            queue_size = roller_queue_od.qsize()
            print(f"Queue size: {queue_size}, Contents: {'Empty' if queue_size == 0 else 'Not Empty'}")

        elif not shared_data["od"]:
            a=False
if __name__ == "__main__":
    PLC_IP = "172.17.8.17"  # Replace with actual PLC IP
    RACK = 0
    SLOT = 1
    DB_NUMBER = 86
    last_detection_time = 0
    DEBOUNCE_INTERVAL = 0.0

    print("Loading YOLO model...")
    model_bigface = YOLO(r"C:\Users\NBC\Desktop\DefectiveRollerDetection\OldModels\Bigfacelatest.pt")
    model_od = YOLO(r"C:\Users\NBC\Desktop\DefectiveRollerDetection\OldModels\ODlatestmodel.pt")


    frame_shape = (960, 1280, 3)


    # Initialize manager and shared data
    manager = Manager()
    shared_data = manager.dict()
    shared_data['bigface'] = False
    shared_data['od'] = False
    shared_data['bigface_presence'] = False
    shared_data['od_presence'] = False

    # Initialize command queue
    command_queue = Queue()

    # Initialize shared memory and variables for multiprocessing
    proximity_count_od = Value('i', 0)
    proximity_count_bigface = Value('i', 0)

    roller_data_od = manager.dict()
    roller_queue_od = Queue()
    roller_queue_bigface = Queue()
    roller_updation_dict = manager.dict()


    # Initialize shared frames
    shared_frame_bigface = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())
    shared_frame_od = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())

    # Initialize locks
    frame_lock_bigface = Lock()
    frame_lock_od = Lock()
    queue_lock = Lock()

    # Start PLC communication process
    plc_process = Process(
        target=plc_communication, 
        args=(PLC_IP, RACK, SLOT, DB_NUMBER, shared_data, command_queue),
        daemon=True
    )
    plc_process.start()

    # Create other processes
    processes = [
        Process(target=capture_frames_bigface, args=(shared_frame_bigface, frame_lock_bigface,frame_shape), daemon=True),
        Process(target=handle_slot_control_bigface, args=(roller_queue_bigface,shared_data,command_queue), daemon=True),
        Process(target=process_rollers_bigface,args=(shared_frame_bigface, frame_lock_bigface, roller_queue_bigface,model_bigface,proximity_count_bigface,roller_updation_dict,queue_lock,shared_data,frame_shape), daemon=True),
        Process(
            target=process_frames_od,
            args=(shared_frame_od, frame_lock_od, roller_data_od, proximity_count_od, roller_queue_od, queue_lock, roller_updation_dict,shared_data,frame_shape),
            daemon=True
        ),
        Process(
            target=capture_frames_od,
            args=(shared_frame_od, frame_lock_od,frame_shape),
            daemon=True
        ),
        Process(
            target=handle_slot_control_od,
            args=(roller_queue_od, roller_data_od,shared_data,command_queue),
            daemon=True
        )
    ]

    # Start all processes
    for process in processes:
        process.start()

    # Main loop to keep processes running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Main: Exiting...")

    # Terminate all processes gracefully
    for process in processes:
        process.terminate()
        process.join()

    plc_process.terminate()
    plc_process.join()