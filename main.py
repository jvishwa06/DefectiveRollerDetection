import os
import sys
import time
import csv
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from multiprocessing import Process, Array, Queue, Lock, Value, Manager
from ultralytics import YOLO
import snap7
from snap7.util import set_bool, get_bool
from snap7.type import Areas

class RollerInspectionGUI:
    def __init__(self, root, shared_data, command_queue, shared_frame_bigface, shared_frame_od, 
                 frame_lock_bigface, frame_lock_od, frame_shape, 
                 shared_annotated_bigface, shared_annotated_od):
        self.root = root

        self.shared_annotated_bigface = shared_annotated_bigface
        self.shared_annotated_od = shared_annotated_od

        self.root.title("Roller Inspection System")
        self.shared_data = shared_data
        self.command_queue = command_queue
        self.shared_frame_bigface = shared_frame_bigface
        self.shared_frame_od = shared_frame_od
        self.frame_lock_bigface = frame_lock_bigface
        self.frame_lock_od = frame_lock_od
        self.frame_shape = frame_shape
        
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.control_panel = ttk.LabelFrame(self.main_container, text="Control Panel")
        self.control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = ttk.Button(self.control_panel, text="Start Inspection", command=self.start_inspection)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(self.control_panel, text="Stop Inspection", command=self.stop_inspection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.camera_frame = ttk.LabelFrame(self.main_container, text="Camera Feeds")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.bigface_label = ttk.Label(self.camera_frame)
        self.bigface_label.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(self.camera_frame, text="Bigface Camera").grid(row=1, column=0)
        
        self.od_label = ttk.Label(self.camera_frame)
        self.od_label.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.camera_frame, text="OD Camera").grid(row=1, column=1)
        
        self.status_frame = ttk.LabelFrame(self.main_container, text="Status")
        self.status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.bigface_status = ttk.Label(self.status_frame, text="Bigface: Waiting")
        self.bigface_status.pack(side=tk.LEFT, padx=10)
        
        self.od_status = ttk.Label(self.status_frame, text="OD: Waiting")
        self.od_status.pack(side=tk.LEFT, padx=10)
        
        self.stats_frame = ttk.LabelFrame(self.main_container, text="Statistics")
        self.stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.total_inspected = ttk.Label(self.stats_frame, text="Total Inspected: 0")
        self.total_inspected.pack(side=tk.LEFT, padx=10)
        
        self.defects_found = ttk.Label(self.stats_frame, text="Defects Found: 0")
        self.defects_found.pack(side=tk.LEFT, padx=10)

        self.good_rollers = ttk.Label(self.stats_frame, text="Good Rollers: 0")
        self.good_rollers.pack(side=tk.LEFT, padx=10)

        self.running = False
        self.processes = []
        self.update_gui()

    def start_inspection(self):
        if not self.running:
            self.running = True
            self.start_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            
            self.processes = [
                Process(target=capture_frames_bigface,args=(self.shared_frame_bigface, self.frame_lock_bigface, self.frame_shape),daemon=True),
                Process(target=handle_slot_control_bigface,args=(roller_queue_bigface, self.shared_data, self.command_queue),daemon=True),
                Process(target=process_rollers_bigface,args=(self.shared_frame_bigface, self.frame_lock_bigface, roller_queue_bigface,model_bigface, proximity_count_bigface, roller_updation_dict,queue_lock, self.shared_data, self.frame_shape),daemon=True),
                Process(target=process_frames_od,args=(self.shared_frame_od, self.frame_lock_od, roller_queue_od, queue_lock,self.shared_data, self.frame_shape, roller_updation_dict),daemon=True),
                Process(target=capture_frames_od,args=(self.shared_frame_od, self.frame_lock_od, self.frame_shape),daemon=True),
                Process(target=handle_slot_control_od,args=(roller_queue_od, self.shared_data, self.command_queue),daemon=True)]
            
            for process in self.processes:
                process.start()

    def stop_inspection(self):
        if self.running:
            self.running = False
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            
            for process in self.processes:
                process.terminate()
                process.join()
            self.processes = []

    def update_gui(self):
        try:
            with self.frame_lock_bigface:
                frame = np.frombuffer(self.shared_annotated_bigface.get_obj(),
                                    dtype=np.uint8).reshape(self.frame_shape)
                frame = cv2.resize(frame, (640, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                self.bigface_label.configure(image=photo)
                self.bigface_label.image = photo

            with self.frame_lock_od:
                frame = np.frombuffer(self.shared_annotated_od.get_obj(),
                                    dtype=np.uint8).reshape(self.frame_shape)
                frame = cv2.resize(frame, (640, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                self.od_label.configure(image=photo)
                self.od_label.image = photo

            self.bigface_status.configure(
                text=f"Bigface: {'Active' if self.shared_data['bigface'] else 'Waiting'}")
            self.od_status.configure(
                text=f"OD: {'Active' if self.shared_data['od'] else 'Waiting'}")


        except Exception as e:
            print(f"GUI Update Error: {e}")

        if not self.root.quit_flag:
            self.root.after(30, self.update_gui)


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
        print(f"PLC Communication: Connection error: {e} ⚠")
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
                print(f"PLC Communication: Error reading sensors: {e} ⚠")

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
                    print(f"PLC Communication: Error handling command: {e} ⚠")

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

        # time.sleep(0.1)
        set_bool(data, byte_index=byte_index, bool_index=bool_index, value=False)
        plc_client.write_area(Areas.DB, db_number, 0, data)
    except Exception as e:
        print(f"PLC Action: Error triggering {action.upper()} slot: {e} ⚠")

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
            annotated_frame = results[0].plot()

            cv2.imwrite(f"{detected_folder}/roller_{pc}.jpg", annotated_frame)

            with frame_lock_bigface:
                np_annotated = np.frombuffer(shared_annotated_bigface.get_obj(), dtype=np.uint8).reshape(frame_shape)
                np.copyto(np_annotated, annotated_frame) 

            defect_detected = any(int(box[-1]) == defect_class_index for box in results[0].boxes.data)

            roller_queue_bigface.put(defect_detected)

            status = "Accepted" if not defect_detected else "Rejected"
            log_bigface_status(f"roller_{pc}", "Defective" if defect_detected else "No Defect", status)
            
            with queue_lock:
                if defect_detected:
                    roller_updation_dict[pc] = 1  # Mark defect detected
                else:
                    roller_updation_dict[pc] = 0  # Mark no defect
            print(f"Roller dict BIGFACE: {roller_updation_dict}")

            queue_list = []
            while not roller_queue_bigface.empty():
                item = roller_queue_bigface.get()
                queue_list.append(item)
            for item in queue_list:
                roller_queue_bigface.put(item)

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

def process_frames_od(shared_frame_od, frame_lock_od, roller_queue_od, queue_lock, shared_data, frame_shape, roller_updation_dict):
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

    model_path = r"best.pt"
    
    yolo = YOLO(model_path).to("cuda")

    try:
        black_frame = np.zeros(frame_shape, dtype=np.uint8)
        yolo.predict(black_frame, device=0, conf=0.3, verbose=False)
        print("Black image YOLO processing for od complete.")
    except Exception as e:
        print(f"Error during YOLO inference on black image: {e}")

    frame_number = 0  # Tracks number of frames captured
    roller_dict = {}  # Stores defect status per roller
    previous_od_state = False
    od_triggered = False
    roller_id_counter = 0  # Unique ID counter for rollers
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
    
                results = yolo.predict(np_frame, device=0, conf=0.3, verbose=False)
                annotated_frame = results[0].plot()
                with frame_lock_od:
                    np_annotated = np.frombuffer(shared_annotated_od.get_obj(), dtype=np.uint8).reshape(frame_shape)
                    np.copyto(np_annotated, annotated_frame)
                
                detections = [
                    ("roller" if int(box[-1]) == 4 else "defect", int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[-1]) , float(box[-2]) )
                    for box in results[0].boxes.data
                ] if results and results[0].boxes.data is not None else []
                
                detections = sorted(detections, key=lambda x: x[1])

                if len(detections) > 0:
                    
                    frame_number += 1

                    save_path = f"{detected_folder}/frame{frame_number}.jpg"
                    cv2.imwrite(save_path, annotated_frame)

                    roller_only_sorted = [detection for detection in detections if detection[0] == "roller"]
                    roller_only_sorted = [ detection for detection in roller_only_sorted if detection[-1] > 0.80 ]

                    defect_only_sorted = [detection for detection in detections if detection[0] == "defect"]

                    for detection in defect_only_sorted:
                        
                        roller_id = point_inside( detection[1:5] , roller_only_sorted , roller_id_counter)

                        if roller_id == 0:
                            print("No defect Found")
                            continue
                        

                        defect_detected =  False if roller_id == 0 else True

                        defect_name = "No Defect" if not defect_detected else yolo.names[detection[5]]

                        print(" found roller_id has defect " , roller_id , " with defect name " , defect_name)

                        # Track roller in dictionary
                        if roller_id in roller_dict:
                            roller_dict[roller_id]['defect'] |= defect_detected  # OR logic
                            roller_dict[roller_id]['defect_names'].append(defect_name)
                        else:
                            roller_dict[roller_id] = {'defect': defect_detected, 'defect_names': [defect_name]}

                    # print("OD Roller Dict",roller_dict)

                    # Track last detected frame
                else:
                    pass
                    # print("No detections found in frame.")
                    # count_of_roller_not_detected += 1
                    # if previous_roller_id_check:
                    #     previous_roller_id = roller_id_counter

                    # if count_of_roller_not_detected > 10:
                    #     roller_id_counter = 0
                    # previous_roller_id_check = False


                if shared_data['bigface'] and not BIGFACE_DETECTED and len(roller_dict) > 0:

                    BIGFACE_DETECTED = True
                
                    print("🚀 again roller dict", roller_dict)


                    defect_detected = list(roller_dict.values())[0]['defect']

                    log_od_status(list(roller_dict.keys())[0],
                                  "Defective" if defect_detected else "No Defect",
                                  "Rejected" if defect_detected else "Accepted",
                                list(roller_dict.values())[0])

                    #"Added Bigface Edge Case Logic- Start"
                    extracted_roller_id_check_bigface = int(list(roller_dict.keys())[0])

                    #"Added Bigface Edge Case Logic- End"

                    first_key = next(iter(roller_dict))  # Get the first key
                    roller_dict.pop(first_key)  # Remove the first key-value pair

                    # print("$$$$$$$$$$$$$$$$$$$$",extracted_roller_id_check_bigface ,roller_updation_dict )

                    if roller_updation_dict[extracted_roller_id_check_bigface] == 0 :

                        with queue_lock:
                            # roller_data_od[roller_id] = defect_detected
                            print("queue check ==> " , defect_detected)
                            roller_queue_od.put(defect_detected)
                            # print_queue_without_emptying(roller_queue_od)
                
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
    last_detection_time = 0
    DEBOUNCE_INTERVAL = 0.0
    initialize_bigface_csv()
    initialize_od_csv()

    print("Loading YOLO model...")
    model_bigface = YOLO(r"bigfacebest.pt")
    model_od = YOLO(r"odbest.pt")

    model_bigface.to('cuda')
    model_od.to('cuda')

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
    shared_annotated_bigface = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())
    shared_annotated_od = Array('B', np.zeros(frame_shape, dtype=np.uint8).flatten())

    frame_lock_bigface = Lock()
    frame_lock_od = Lock()
    queue_lock = Lock()

    plc_process = Process(target=plc_communication, args=(PLC_IP, RACK, SLOT, DB_NUMBER, shared_data, command_queue),daemon=True)
    plc_process.start()
    try:
        root = tk.Tk()
        root.quit_flag = False  

        gui = RollerInspectionGUI(
            root, shared_data, command_queue,
            shared_frame_bigface, shared_frame_od,
            frame_lock_bigface, frame_lock_od,
            frame_shape, shared_annotated_bigface,
            shared_annotated_od
        )

        def on_closing():
            """Handle window closing event"""
            root.quit_flag = True
            if gui.running:
                gui.stop_inspection()
            plc_process.terminate() 
            plc_process.join()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    except KeyboardInterrupt:
        print("Main: Keyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"Main: Error occurred: {e}")
    finally:
        if 'gui' in locals() and gui.running:
            gui.stop_inspection()
        if 'plc_process' in locals() and plc_process.is_alive():
            plc_process.terminate()
            plc_process.join()

    processes = [
        Process(target=capture_frames_bigface, args=(shared_frame_bigface, frame_lock_bigface,frame_shape), daemon=True),
        Process(target=handle_slot_control_bigface, args=(roller_queue_bigface,shared_data,command_queue), daemon=True),
        Process(target=process_rollers_bigface,args=(shared_frame_bigface, frame_lock_bigface, roller_queue_bigface,model_bigface,proximity_count_bigface,roller_updation_dict,queue_lock,shared_data,frame_shape), daemon=True),
        Process(target=process_frames_od,args=(shared_frame_od, frame_lock_od, roller_queue_od, queue_lock, shared_data, frame_shape, roller_updation_dict),daemon=True),
        Process(target=capture_frames_od,args=(shared_frame_od, frame_lock_od,frame_shape),daemon=True),
        Process(target=handle_slot_control_od,args=(roller_queue_od,shared_data,command_queue),daemon=True)]

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