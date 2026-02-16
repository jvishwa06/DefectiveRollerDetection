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

def initialize_bigface_csv():
    """Initialize the Bigface CSV file with headers."""
    if not os.path.exists("bigface_defects_log.csv"):
        with open("bigface_defects_log.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["roller_id", "defect_status", "plc status"])

def log_bigface_status(roller_id, defect_status, status):
    """Log the defect status of a roller in the Bigface CSV."""
    with open("bigface_defects_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([roller_id, defect_status, status])

def initialize_od_csv():
    """Initialize the od CSV file with headers."""
    if not os.path.exists("OD_defects_log.csv"):
        with open("od_defects_log.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["roller_id", "defect_status", "plc status", "od_dictionary"])

def log_od_status(roller_id, defect_status, status, od_dictionary):
    """Log the defect status of a roller in the OD CSV."""
    with open("od_defects_log.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([roller_id, defect_status, status, od_dictionary])


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
        data = bytearray(2)
        set_bool(data, byte_index=byte_index, bool_index=bool_index, value=True)
        plc_client.write_area(Areas.DB, db_number, 0, data)

        # time.sleep(0.1)
        set_bool(data, byte_index=byte_index, bool_index=bool_index, value=False)
        plc_client.write_area(Areas.DB, db_number, 0, data)
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

def process_rollers_bigface(shared_frame_bigface, frame_lock_bigface, roller_queue_bigface, model_bigface, proximity_count_bigface, roller_updation_dict, queue_lock, shared_data, frame_shape):
    """Process frames for YOLO inference."""
    detected_folder = "captured_bigface_frames"
    os.makedirs(detected_folder, exist_ok=True)

    black_frame = np.zeros(frame_shape, dtype=np.uint8)
    try:
        results = model_bigface.predict(black_frame, device=0, conf=0.3, verbose=False)
        print("Black image YOLO processing for bigface complete.")
    except Exception as e:
        print(f"Error during YOLO inference on black image: {e}")

    roller_detected = False
    
    while True:
        if shared_data["bigface_presence"] and not roller_detected:
            roller_detected = True
            print("Roller detected in bigface. Capturing frame...")

            with frame_lock_bigface:
                np_frame = np.frombuffer(shared_frame_bigface.get_obj(), dtype=np.uint8).reshape(frame_shape)
                frame = np_frame.copy()

            proximity_count_bigface.value += 1
            pc = proximity_count_bigface.value

            defect_class_index = next((key for key, value in model_bigface.names.items() if value == 'damage'), None)
            if defect_class_index is None:
                print("Defect class 'damage' not found in model.")
                continue

            results = model_bigface.predict(frame, device=0, conf=0.3, verbose=False)

            cv2.imwrite(f"{detected_folder}/roller_{pc}.jpg", results[0].plot())

            defect_detected = any(int(box[-1]) == defect_class_index for box in results[0].boxes.data)

            roller_queue_bigface.put(defect_detected)

            status = "Accepted" if not defect_detected else "Rejected"
            log_bigface_status(f"roller_{pc}", "Defective" if defect_detected else "No Defect", status)
            
            with queue_lock:
                if defect_detected:
                    roller_updation_dict[pc] = 1  
                else:
                    roller_updation_dict[pc] = 0  
            print(f"Roller dict BIGFACE: {roller_updation_dict}")

            queue_list = []
            while not roller_queue_bigface.empty():
                item = roller_queue_bigface.get()
                queue_list.append(item)
            for item in queue_list:
                roller_queue_bigface.put(item)

        elif not shared_data['bigface_presence']:
            roller_detected = False

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
                print(f"Slot control received roller status for Bigface: {status}")
                command_queue.put(("accept_bigface" if not defect_detected else "reject_bigface", None))
        elif not shared_data["bigface"]:
            a = False


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

def process_frames_od(shared_frame_od, frame_lock_od, roller_queue_od, queue_lock, shared_data, frame_shape, roller_updation_dict, model_od):
    """Process frames for YOLO inference and track roller defects with pulse debounce & proper exit handling."""

    detected_folder = "captured_od_frames"
    os.makedirs(detected_folder, exist_ok=True)

    def point_inside(rectangle, list_of_all_rollers , roller_number):
        
        length = len(list_of_all_rollers)

        check = 3 if roller_number > 3 else roller_number

        if length > check:
            length =- 1

        roller_dictioanry = { i : list_of_all_rollers[idx] for idx,i in enumerate(range(roller_number , roller_number - length , -1)) }
        print("roller_dictioanry in  inside point ----> " , roller_dictioanry , " roller_number :" , roller_number , " length :" , length)

        for idx,roller in roller_dictioanry.items():
            entire_coordinates = roller[1:5]    
            x1, y1 ,x2, y2  = [ int(i) for i in rectangle]
            decision = (entire_coordinates[0] <= x1 <= entire_coordinates[2] and entire_coordinates[1] <= y1 <= entire_coordinates[3]) or (entire_coordinates[0] <= x2 <= entire_coordinates[2] and entire_coordinates[1] <= y2 <= entire_coordinates[3])
            if decision:
                return idx
        return 0
    

    try:
        black_frame = np.zeros(frame_shape, dtype=np.uint8)
        model_od.predict(black_frame, device=0, conf=0.9, verbose=False)
        print("Black image YOLO processing for od complete.")
    except Exception as e:
        print(f"Error during YOLO inference on black image: {e}")

    frame_number = 0  
    roller_dict = {}  
    previous_od_state = False
    od_triggered = False
    roller_id_counter = 0  
    BIGFACE_DETECTED = False

    while True:
        current_od_state = shared_data["od_presence"]

        if current_od_state and not previous_od_state:
            od_triggered = True

            roller_id_counter += 1
            
            roller_dict[roller_id_counter] = {'defect': False , 'defect_names': ["No defect"]}

            print(f"\n🎯 New roller detected! Assigned Roller ID: {roller_id_counter} , in frame number : {frame_number + 1}")
            
            
        if od_triggered:
                
                with frame_lock_od:
                    np_frame = np.frombuffer(shared_frame_od.get_obj(), dtype=np.uint8).reshape(frame_shape)
    
                results = model_od.predict(np_frame, device=0, conf=0.3, save = True ,verbose=False)
                plot_image = results[0].plot() 

                detections = [
                    ("roller" if int(box[-1]) == 4 else "defect", int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[-1]) , float(box[-2]) )
                    for box in results[0].boxes.data
                ] if results and results[0].boxes.data is not None else []
                
                detections = sorted(detections, key=lambda x: x[1])  # Sort by x-coordinate

                if len(detections) > 0:
                    
                    frame_number += 1

                    save_path = f"{detected_folder}/frame{frame_number}.jpg"
                    cv2.imwrite(save_path, plot_image)

                    roller_only_sorted = [detection for detection in detections if detection[0] == "roller"]
                    roller_only_sorted = [ detection for detection in roller_only_sorted if detection[-1] > 0.60 ]

                    defect_only_sorted = [detection for detection in detections if detection[0] == "defect"]

                    for detection in defect_only_sorted:
                        
                        roller_id = point_inside( detection[1:5] , roller_only_sorted , roller_id_counter)

                        if roller_id == 0:
                            print("No defect Found")
                            continue
                        

                        defect_detected =  False if roller_id == 0 else True

                        defect_name = "No Defect" if not defect_detected else model_od.names[detection[5]]

                        print(" found roller_id has defect " , roller_id , " with defect name " , defect_name)

                        if roller_id in roller_dict:
                            roller_dict[roller_id]['defect'] |= defect_detected  # OR logic
                            roller_dict[roller_id]['defect_names'].append(defect_name)
                        else:
                            roller_dict[roller_id] = {'defect': defect_detected, 'defect_names': [defect_name]}


                if shared_data['bigface'] and not BIGFACE_DETECTED and len(roller_dict) > 0:

                    BIGFACE_DETECTED = True
                
                    print("🚀 again roller dict", roller_dict)


                    defect_detected = list(roller_dict.values())[0]['defect']

                    log_od_status(list(roller_dict.keys())[0],
                                  "Defective" if defect_detected else "No Defect",
                                  "Rejected" if defect_detected else "Accepted",
                                list(roller_dict.values())[0])

                    extracted_roller_id_check_bigface = int(list(roller_dict.keys())[0])

                    first_key = next(iter(roller_dict)) 
                    roller_dict.pop(first_key)  

                    if roller_updation_dict[extracted_roller_id_check_bigface] == 1 :

                        with queue_lock:
                            print("queue check ==> " , defect_detected)
                            roller_queue_od.put(defect_detected)
                
                elif not shared_data['bigface']:
                    BIGFACE_DETECTED = False

        previous_od_state = current_od_state


def handle_slot_control_od(roller_queue_od, shared_data, command_queue):
    """Control slot mechanism based on second proximity sensor."""
    processing = False
    while True:
        if shared_data["od"] and not processing and not roller_queue_od.empty():
            processing = True
            if not roller_queue_od.empty():
                defect_detected = roller_queue_od.get()
                status = "❌ Defective" if defect_detected else "✅ Good"
                print(f"Slot control received roller status for od: {status}")
                command_queue.put(("reject_od" if defect_detected else "accept_od" , None))

            queue_size = roller_queue_od.qsize()
            print(f"📌 Queue size: {queue_size}, Contents: {'Empty' if queue_size == 0 else 'Not Empty'}")

        elif not shared_data["od"]:
            processing = False


if __name__ == "__main__":
    PLC_IP = "172.17.8.17" 
    RACK = 0
    SLOT = 1
    DB_NUMBER = 86
    initialize_bigface_csv()
    initialize_od_csv()

    print("Loading YOLO model...")
    model_bigface = YOLO(r"OldModels\Bigfacelatest.pt")
    model_bigface.to('cuda')

    frame_shape = (960, 1280, 3)

    manager = Manager()
    shared_data = manager.dict()
    shared_data['bigface'] = False
    shared_data['od'] = False
    shared_data['bigface_presence'] = False
    shared_data['od_presence'] = False

    command_queue = Queue()

    proximity_count_od = Value('i', 0)
    proximity_count_bigface = Value('i', 0)

    roller_data_od = manager.dict()
    roller_queue_od = Queue()
    roller_queue_bigface = Queue()
    roller_updation_dict = manager.dict()


    shared_frame_bigface = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())
    shared_frame_od = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())

    frame_lock_bigface = Lock()
    frame_lock_od = Lock()
    queue_lock = Lock()

    plc_process = Process(target=plc_communication, args=(PLC_IP, RACK, SLOT, DB_NUMBER, shared_data, command_queue),daemon=True)
    plc_process.start()

    processes = [
        Process(target=capture_frames_bigface, args=(shared_frame_bigface, frame_lock_bigface,frame_shape), daemon=True),
        Process(target=handle_slot_control_bigface, args=(roller_queue_bigface,shared_data,command_queue), daemon=True),
        Process(target=process_rollers_bigface,args=(shared_frame_bigface, frame_lock_bigface, roller_queue_bigface,model_bigface,proximity_count_bigface,roller_updation_dict,queue_lock,shared_data,frame_shape), daemon=True),
        Process(target=process_frames_od,args=(shared_frame_od, frame_lock_od, roller_queue_od, queue_lock, shared_data, frame_shape, roller_updation_dict, model_od),daemon=True),
        Process(target=capture_frames_od,args=(shared_frame_od, frame_lock_od,frame_shape),daemon=True),
        Process(target=handle_slot_control_od,args=(roller_queue_od,shared_data,command_queue),daemon=True)
        ]

    for process in processes:
        process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Main: Exiting...")

    for process in processes:
        process.terminate()
        process.join()

    plc_process.terminate()
    plc_process.join()