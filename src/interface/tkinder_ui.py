import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import threading
import time
from multiprocessing import Process, Array, Queue, Lock, Value, Manager
from ultralytics import YOLO
from main import *
import snap7
from snap7.util import set_bool
from snap7.type import Areas


users = {
    "": {"password": "", "role": "User"},
    "admin@welvision.com": {"password": "admin123", "role": "Admin"},
    "superadmin@welvision.com": {"password": "super123", "role": "Super Admin"}
}

class WelVisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WELVISION")
        self.geometry("1366x768")
        self.configure(bg="#0a2158")
        self.iconbitmap(default="") 
        
        # Center window on screen
        self.center_window()
        
        # Initialize variables
        self.current_user = None
        self.current_role = None
        
        # Statistics variables
        self.od_inspected = 0
        self.od_defective = 0
        self.od_good = 0
        
        self.bf_inspected = 0
        self.bf_defective = 0
        self.bf_good = 0
        
        # Camera variables
        self.od_camera = None
        self.bf_camera = None
        self.od_frame = None
        self.bf_frame = None
        
        # Inspection status
        self.inspection_running = False
        
        # Defect thresholds - Updated to match user requirements
        self.od_defect_thresholds = {
            "Rust": 50,
            "Dent": 50,
            "Spherical Mark": 50,
            "Damage": 50,
            "Flat Line": 50,
            "Damage on End": 50,
            "Roller": 50
        }
        
        self.bf_defect_thresholds = {
            "Damage": 50,
            "Rust": 50,
            "Dent": 50,
            "Roller": 50
        }
        
        # Show login page
        self.show_login_page()


    
    def center_window(self):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 1280) // 2
        y = (screen_height - 800) // 2
        self.geometry(f"1280x800+{x}+{y}")
    
    def show_login_page(self):
        # Clear any existing widgets
        for widget in self.winfo_children():
            widget.destroy()
        
        # Create login frame with black background
        login_frame = tk.Frame(self, bg="black", width=500, height=600)
        login_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Logo
        logo_label = tk.Label(login_frame, text="WELVISION", font=("Arial", 32, "bold"), fg="white", bg="black")
        logo_label.pack(pady=(0, 20))
        
        # Subtitle
        subtitle_label = tk.Label(login_frame, text="Please sign in to continue", font=("Arial", 14), fg="white", bg="black")
        subtitle_label.pack(pady=(0, 30))
        
        # Role selection
        role_frame = tk.Frame(login_frame, bg="black")
        role_frame.pack(pady=(0, 20))
        
        self.role_var = tk.StringVar(value="User")
        
        user_rb = tk.Radiobutton(role_frame, text="User", variable=self.role_var, value="User", 
                                font=("Arial", 12), fg="white", bg="black", selectcolor="black")
        admin_rb = tk.Radiobutton(role_frame, text="Admin", variable=self.role_var, value="Admin", 
                                 font=("Arial", 12), fg="white", bg="black", selectcolor="black")
        super_admin_rb = tk.Radiobutton(role_frame, text="Super Admin", variable=self.role_var, value="Super Admin", 
                                       font=("Arial", 12), fg="white", bg="black", selectcolor="black")
        
        user_rb.pack(side=tk.LEFT, padx=10)
        admin_rb.pack(side=tk.LEFT, padx=10)
        super_admin_rb.pack(side=tk.LEFT, padx=10)
        
        # Email
        email_label = tk.Label(login_frame, text="Email", font=("Arial", 12), fg="white", bg="black", anchor="w")
        email_label.pack(fill="x", pady=(0, 5))
        
        self.email_entry = tk.Entry(login_frame, font=("Arial", 12), width=40)
        self.email_entry.pack(pady=(0, 15), ipady=8)
        
        # Password
        password_label = tk.Label(login_frame, text="Password", font=("Arial", 12), fg="white", bg="black", anchor="w")
        password_label.pack(fill="x", pady=(0, 5))
        
        self.password_entry = tk.Entry(login_frame, font=("Arial", 12), width=40, show="*")
        self.password_entry.pack(pady=(0, 30), ipady=8)
        
        # Sign in button
        sign_in_button = tk.Button(login_frame, text="Sign In", font=("Arial", 12, "bold"), 
                                  bg="#007bff", fg="white", width=20, height=2, 
                                  command=self.authenticate)
        sign_in_button.pack(pady=10)
        
        # Bind Enter key to authenticate
        self.bind("<Return>", lambda event: self.authenticate())
    
    def authenticate(self):
        email = self.email_entry.get().strip()
        password = self.password_entry.get().strip()
        role = self.role_var.get()
        
        if email in users and users[email]["password"] == password and users[email]["role"] == role:
            self.current_user = email
            self.current_role = role
            self.show_main_interface()
        else:
            messagebox.showerror("Login Failed", "Invalid email, password, or role.")

    def show_main_interface(self):
        # Clear any existing widgets
        self.initialize_system()
        for widget in self.winfo_children():
            widget.destroy()
        
        # Create main frame
        main_frame = tk.Frame(self, bg="#0a2158")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_frame = tk.Frame(main_frame, bg="#0a2158", height=50)
        header_frame.pack(fill=tk.X)
        
        # Logo in header
        logo_label = tk.Label(header_frame, text="WELVISION", font=("Arial", 18, "bold"), fg="white", bg="#0a2158")
        logo_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        # User info
        user_label = tk.Label(header_frame, text=f"{self.current_role}: {self.current_user}", 
                             font=("Arial", 12), fg="white", bg="#0a2158")
        user_label.pack(side=tk.RIGHT, padx=20, pady=5)
        
        # Logout button
        logout_button = tk.Button(header_frame, text="Logout", font=("Arial", 10), 
                                 command=self.show_login_page)
        logout_button.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Create tabs with custom style
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook.Tab', background="#0a2158", foreground="white", 
                       font=('Arial', 12, 'bold'), padding=[20, 10], borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', '#1a3168')], 
                 foreground=[('selected', 'white')])
        style.configure('TNotebook', background="#0a2158", borderwidth=0)
        
        tab_control = ttk.Notebook(main_frame)
        
        inference_tab = tk.Frame(tab_control, bg="#0a2158")
        statistics_tab = tk.Frame(tab_control, bg="#0a2158")
        settings_tab = tk.Frame(tab_control, bg="#0a2158")
        
        tab_control.add(inference_tab, text="Inference")
        tab_control.add(statistics_tab, text="Statistics")
        tab_control.add(settings_tab, text="Settings")
        
        tab_control.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Configure tabs
        self.setup_inference_tab(inference_tab)
        self.setup_statistics_tab(statistics_tab)
        self.setup_settings_tab(settings_tab)
        
        # Start camera feeds
        self.start_camera_feeds()
        
        # Start updating statistics
        self.update_statistics()
    

    # Function to handle model confidence adjustments
    def update_model_confidence(self):
        """Updates the confidence thresholds of the YOLO models in real-time."""
        if not hasattr(self, 'inspection_running') or not self.inspection_running:
            return
        
        # Get current confidence values from sliders
        od_conf = self.od_conf_threshold
        bf_conf = self.bf_conf_threshold
        
        print(f"Updating model confidence: OD={od_conf:.2f}, Bigface={bf_conf:.2f}")
        
        # Update the shared data dictionary with new confidence values
        if hasattr(self, 'shared_data'):
            self.shared_data['od_conf_threshold'] = od_conf
            self.shared_data['bf_conf_threshold'] = bf_conf
        
        # If you need to update the models directly (if they're in the main process)
        if hasattr(self, 'model_od'):
            self.model_od.conf = od_conf
        
        if hasattr(self, 'model_bigface'):
            self.model_bigface.conf = bf_conf

    def setup_settings_tab(self, parent):
        """
        Creates the complete UI for the settings tab, including model confidence sliders
        and defect threshold controls.
        
        Args:
            parent: The parent frame (settings tab)
        """
        # Main container for settings
        settings_container = tk.Frame(parent, bg="#0a2158")
        settings_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Settings Title
        title_label = tk.Label(settings_container, text="Model & Defect Settings", 
                            font=("Arial", 18, "bold"), fg="white", bg="#0a2158")
        title_label.pack(pady=(0, 20))
        
        # ===== MODEL CONFIDENCE SECTION =====
        conf_frame = tk.LabelFrame(settings_container, text="Model Confidence Thresholds", 
                                font=("Arial", 14, "bold"), fg="white", bg="#0a2158", bd=2)
        conf_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # OD Model Confidence Slider
        od_conf_frame = tk.Frame(conf_frame, bg="#0a2158", pady=10)
        od_conf_frame.pack(fill=tk.X, padx=10)
        
        od_conf_label = tk.Label(od_conf_frame, text="OD Model Confidence", font=("Arial", 12), 
                            fg="white", bg="#0a2158", width=20, anchor="w")
        od_conf_label.pack(side=tk.LEFT, padx=10)
        
        # Initialize with default value (25% = 0.25)
        if not hasattr(self, 'od_conf_threshold'):
            self.od_conf_threshold = 0.25
        self.od_conf_slider_value = tk.DoubleVar(value=self.od_conf_threshold * 100)
        
        od_conf_slider = ttk.Scale(od_conf_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                length=300, variable=self.od_conf_slider_value)
        od_conf_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.od_conf_value_label = tk.Label(od_conf_frame, text=f"{int(self.od_conf_threshold * 100)}%", 
                                        font=("Arial", 12), fg="white", bg="#0a2158", width=5)
        self.od_conf_value_label.pack(side=tk.LEFT, padx=10)
        
        # Update OD confidence value label when slider is moved
        def update_od_conf_label(val):
            self.od_conf_value_label.config(text=f"{int(float(val))}%")
            # Update the actual threshold value
            self.od_conf_threshold = float(val) / 100
            # If inspection is running, update the model in real-time
            if hasattr(self, 'inspection_running') and self.inspection_running:
                self.update_model_confidence()
        
        od_conf_slider.config(command=update_od_conf_label)
        
        # BIG FACE Model Confidence Slider
        bf_conf_frame = tk.Frame(conf_frame, bg="#0a2158", pady=10)
        bf_conf_frame.pack(fill=tk.X, padx=10)
        
        bf_conf_label = tk.Label(bf_conf_frame, text="Bigface Model Confidence", font=("Arial", 12), 
                            fg="white", bg="#0a2158", width=20, anchor="w")
        bf_conf_label.pack(side=tk.LEFT, padx=10)
        
        # Initialize with default value (25% = 0.25)
        if not hasattr(self, 'bf_conf_threshold'):
            self.bf_conf_threshold = 0.25
        self.bf_conf_slider_value = tk.DoubleVar(value=self.bf_conf_threshold * 100)
        
        bf_conf_slider = ttk.Scale(bf_conf_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                length=300, variable=self.bf_conf_slider_value)
        bf_conf_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.bf_conf_value_label = tk.Label(bf_conf_frame, text=f"{int(self.bf_conf_threshold * 100)}%", 
                                        font=("Arial", 12), fg="white", bg="#0a2158", width=5)
        self.bf_conf_value_label.pack(side=tk.LEFT, padx=10)
        
        # Update BF confidence value label when slider is moved
        def update_bf_conf_label(val):
            self.bf_conf_value_label.config(text=f"{int(float(val))}%")
            # Update the actual threshold value
            self.bf_conf_threshold = float(val) / 100
            # If inspection is running, update the model in real-time
            if hasattr(self, 'inspection_running') and self.inspection_running:
                self.update_model_confidence()
        
        bf_conf_slider.config(command=update_bf_conf_label)
        
        # Separator between sections
        separator = ttk.Separator(settings_container, orient="horizontal")
        separator.pack(fill=tk.X, padx=10, pady=20)
        
        # ===== DEFECT THRESHOLDS SECTION =====
        # Frame for OD Thresholds
        od_thresholds_frame = tk.LabelFrame(settings_container, text="OD Defect Thresholds", 
                                        font=("Arial", 14, "bold"), fg="white", bg="#0a2158", bd=2)
        od_thresholds_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        # Create a scrollable frame for OD thresholds if there are many
        od_canvas = tk.Canvas(od_thresholds_frame, bg="#0a2158", highlightthickness=0)
        od_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        od_scrollbar = ttk.Scrollbar(od_thresholds_frame, orient="vertical", command=od_canvas.yview)
        od_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        od_canvas.configure(yscrollcommand=od_scrollbar.set)
        od_canvas.bind('<Configure>', lambda e: od_canvas.configure(scrollregion=od_canvas.bbox("all")))
        
        od_sliders_frame = tk.Frame(od_canvas, bg="#0a2158")
        od_canvas.create_window((0, 0), window=od_sliders_frame, anchor="nw")
        
        # Create sliders for OD defects
        row = 0
        for defect, value in self.od_defect_thresholds.items():
            self.create_slider(od_sliders_frame, defect, 0, 100, value, row, is_od=True)
            row += 1
        
        # Frame for BIG FACE Thresholds
        bf_thresholds_frame = tk.LabelFrame(settings_container, text="BIG FACE Defect Thresholds", 
                                        font=("Arial", 14, "bold"), fg="white", bg="#0a2158", bd=2)
        bf_thresholds_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        # Create a scrollable frame for BF thresholds if there are many
        bf_canvas = tk.Canvas(bf_thresholds_frame, bg="#0a2158", highlightthickness=0)
        bf_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        bf_scrollbar = ttk.Scrollbar(bf_thresholds_frame, orient="vertical", command=bf_canvas.yview)
        bf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        bf_canvas.configure(yscrollcommand=bf_scrollbar.set)
        bf_canvas.bind('<Configure>', lambda e: bf_canvas.configure(scrollregion=bf_canvas.bbox("all")))
        
        bf_sliders_frame = tk.Frame(bf_canvas, bg="#0a2158")
        bf_canvas.create_window((0, 0), window=bf_sliders_frame, anchor="nw")
        
        # Create sliders for BIG FACE defects
        row = 0
        for defect, value in self.bf_defect_thresholds.items():
            self.create_slider(bf_sliders_frame, defect, 0, 100, value, row, is_od=False)
            row += 1
        
        # Save button
        save_button = tk.Button(settings_container, text="Save Settings", font=("Arial", 12, "bold"),
                            bg="#28a745", fg="white", command=self.save_thresholds)
        save_button.pack(pady=20)
        
        # Make sure the update_model_confidence method exists
        if not hasattr(self, 'update_model_confidence'):
            def update_model_confidence(self):
                """
                Updates the confidence thresholds of the YOLO models in real-time.
                This function sends the updated confidence values to the processing functions.
                """
                if not hasattr(self, 'inspection_running') or not self.inspection_running:
                    return
                
                # Get current confidence values from sliders
                od_conf = self.od_conf_threshold
                bf_conf = self.bf_conf_threshold
                
                print(f"Updating model confidence: OD={od_conf:.2f}, Bigface={bf_conf:.2f}")
                
                # Update the shared data dictionary with new confidence values
                if hasattr(self, 'shared_data'):
                    self.shared_data['od_conf_threshold'] = od_conf
                    self.shared_data['bf_conf_threshold'] = bf_conf
                
                # If you need to update the models directly (if they're in the main process)
                if hasattr(self, 'model_od'):
                    self.model_od.conf = od_conf
                
                if hasattr(self, 'model_bigface'):
                    self.model_bigface.conf = bf_conf
            
            self.update_model_confidence = update_model_confidence
        
        # Make sure the save_thresholds method exists
        if not hasattr(self, 'save_thresholds'):
            def save_thresholds(self):
                """
                Saves the current threshold settings.
                """
                # Save model confidence thresholds
                self.od_conf_threshold = float(self.od_conf_slider_value.get()) / 100
                self.bf_conf_threshold = float(self.bf_conf_slider_value.get()) / 100
                
                # Save defect thresholds
                for defect in self.od_defect_thresholds:
                    key = f"od_{defect}"
                    if key in self.slider_values:
                        self.od_defect_thresholds[defect] = int(float(self.slider_values[key].get()))
                
                for defect in self.bf_defect_thresholds:
                    key = f"bf_{defect}"
                    if key in self.slider_values:
                        self.bf_defect_thresholds[defect] = int(float(self.slider_values[key].get()))
                
                # Update the models with new confidence values
                self.update_model_confidence()
                
                messagebox.showinfo("Settings Saved", "Model confidence and defect thresholds have been saved successfully.")
                print(f"OD Confidence: {self.od_conf_threshold}, BF Confidence: {self.bf_conf_threshold}")
                print(f"OD Thresholds: {self.od_defect_thresholds}")
                print(f"BF Thresholds: {self.bf_defect_thresholds}")
            
            self.save_thresholds = save_thresholds

    def setup_inference_tab(self, parent):
        # Create frames for camera feeds
        camera_frame = tk.Frame(parent, bg="#0a2158")
        camera_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=10)
        
        # OD Camera frame
        od_frame = tk.LabelFrame(camera_frame, text="Camera 1 - OD", font=("Arial", 12, "bold"), 
                                fg="white", bg="#0a2158", bd=2)
        od_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        self.od_canvas = tk.Canvas(od_frame, bg="black", width=400, height=250)
        self.od_canvas.pack(padx=10, pady=5)
        
        # BIG FACE Camera frame
        bf_frame = tk.LabelFrame(camera_frame, text="Camera 2 - BIG FACE", font=("Arial", 12, "bold"), 
                                fg="white", bg="#0a2158", bd=2)
        bf_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        self.bf_canvas = tk.Canvas(bf_frame, bg="black", width=400, height=250)
        self.bf_canvas.pack(padx=10, pady=5)
        
        # Configure grid weights
        camera_frame.grid_columnconfigure(0, weight=1)
        camera_frame.grid_columnconfigure(1, weight=1)
        camera_frame.grid_rowconfigure(0, weight=1)
        
        # Control buttons frame
        control_buttons_frame = tk.Frame(parent, bg="#0a2158")
        control_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_button = tk.Button(control_buttons_frame, text="Start Inspection", font=("Arial", 10, "bold"), 
                                    bg="#28a745", fg="white", width=12, height=1, 
                                    command=self.start_inspection)
        self.start_button.pack(side=tk.LEFT, padx=20, pady=5)
        
        self.stop_button = tk.Button(control_buttons_frame, text="Stop Inspection", font=("Arial", 10, "bold"), 
                                    bg="#dc3545", fg="white", width=12, height=1, 
                                    command=self.stop_inspection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Defect threshold controls
        threshold_panel = tk.LabelFrame(parent, text="Defect Threshold Controls", font=("Arial", 14, "bold"), 
                                      fg="white", bg="#0a2158", bd=2)
        threshold_panel.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        # Create a frame for OD thresholds
        od_threshold_frame = tk.LabelFrame(threshold_panel, text="OD Defect Thresholds", font=("Arial", 12, "bold"), 
                                         fg="white", bg="#0a2158", bd=1)
        od_threshold_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create a scrollable frame for OD thresholds
        od_canvas = tk.Canvas(od_threshold_frame, bg="#0a2158", highlightthickness=0)
        od_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        od_scrollbar = ttk.Scrollbar(od_threshold_frame, orient="vertical", command=od_canvas.yview)
        od_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        od_canvas.configure(yscrollcommand=od_scrollbar.set)
        od_canvas.bind('<Configure>', lambda e: od_canvas.configure(scrollregion=od_canvas.bbox("all")))
        
        od_sliders_frame = tk.Frame(od_canvas, bg="#0a2158")
        od_canvas.create_window((0, 0), window=od_sliders_frame, anchor="nw")
        
        # Create sliders for OD defects
        row = 0
        for defect, value in self.od_defect_thresholds.items():
            self.create_slider(od_sliders_frame, defect, 0, 100, value, row)
            row += 1
        
        # Create a frame for BIG FACE thresholds
        bf_threshold_frame = tk.LabelFrame(threshold_panel, text="BIG FACE Defect Thresholds", font=("Arial", 12, "bold"), 
                                         fg="white", bg="#0a2158", bd=1)
        bf_threshold_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Create sliders for BIG FACE defects
        row = 0
        for defect, value in self.bf_defect_thresholds.items():
            self.create_slider(bf_threshold_frame, defect, 0, 100, value, row)
            row += 1
        
        # Configure grid weights for threshold panel
        threshold_panel.grid_columnconfigure(0, weight=1)
        threshold_panel.grid_columnconfigure(1, weight=1)


    def create_processes(self):
        """Recreates process instances before starting them."""
        self.plc_process = Process(
            target=plc_communication,
            args=(self.PLC_IP, self.RACK, self.SLOT, self.DB_NUMBER, self.shared_data, self.command_queue),
            daemon=True
        )

        self.processes = [
            Process(target=capture_frames_bigface, args=(self.shared_frame_bigface, self.frame_lock_bigface, self.frame_shape), daemon=True),
            Process(target=handle_slot_control_bigface, args=(self.roller_queue_bigface, self.shared_data, self.command_queue), daemon=True),
            Process(target=process_rollers_bigface, args=(self.shared_frame_bigface, self.frame_lock_bigface, self.roller_queue_bigface, self.model_bigface, self.proximity_count_bigface, self.roller_updation_dict, self.queue_lock, self.shared_data, self.frame_shape, self.shared_annotated_bigface, self.annotated_frame_lock_bigface), daemon=True),
            Process(target=process_frames_od, args=(self.shared_frame_od, self.frame_lock_od, self.roller_queue_od, self.queue_lock, self.shared_data, self.frame_shape, self.roller_updation_dict, self.shared_annotated_od, self.annotated_frame_lock_od), daemon=True),
            Process(target=capture_frames_od, args=(self.shared_frame_od, self.frame_lock_od, self.frame_shape), daemon=True),
            Process(target=handle_slot_control_od, args=(self.roller_queue_od, self.shared_data, self.command_queue), daemon=True)
        ]



    def start_inspection(self):
        if self.inspection_running:
            print("Inspection is already running!")
            return

        self.inspection_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Recreate processes before starting
        self.create_processes()

        # Start PLC process
        if self.plc_process is not None:
            self.plc_process.start()

        # Start subprocesses
        for process in self.processes:
            process.start()

        

        

    def stop_inspection(self):
        """Stops all running processes."""
        if not self.inspection_running:
            print("Inspection is not running.")
            return

                # PLC connection details
        PLC_IP = "172.17.8.17"
        RACK = 0
        SLOT = 1
        DB_NUMBER = 86

        # Create PLC client
        plc_client = snap7.client.Client()

        try:
            plc_client.connect(PLC_IP, RACK, SLOT)
            print("✅ PLC Communication: Connected to PLC.")
        except Exception as e:
            print(f"❌ PLC Communication: Failed to connect to PLC. Error: {e}")
            exit()

        # Read existing DB data (Ensure persistence)
        data = plc_client.read_area(Areas.DB, DB_NUMBER, 0, 2)  # Read 2 bytes
        print(f"🔍 Existing Data: {list(data)}")

        # Modify specific bits while keeping other bits unchanged
        set_bool(data, byte_index=1, bool_index=6, value=False)  # Turn ON specific bit
        set_bool(data, byte_index=1, bool_index=7, value=False)  # Turn ON another bit

        # Write back the modified data to DB
        plc_client.write_area(Areas.DB, DB_NUMBER, 0, data)
        print("✅ PLC Communication: Lights ON signal sent with persistence.")
        # Close PLC connection
        plc_client.disconnect()
                
        self.inspection_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Stop the PLC process if it's running
        if self.plc_process.is_alive():
            self.plc_process.terminate()
            self.plc_process.join()
            self.plc_process = None  # Mark it for recreation

        # Stop and clear all subprocesses
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join()

        self.processes = []  # Clear the list of processes

        print("Inspection stopped.")

    def save_thresholds(self):
        """Saves the current threshold settings."""
        # Save model confidence thresholds
        self.od_conf_threshold = float(self.od_conf_slider_value.get()) / 100
        self.bf_conf_threshold = float(self.bf_conf_slider_value.get()) / 100
        
        # Print the updated values
        print(f"OD Confidence: {self.od_conf_threshold}, BF Confidence: {self.bf_conf_threshold}")
        
        # Update the models with new confidence values
        self.update_model_confidence()
        
        messagebox.showinfo("Settings Saved", "Model confidence settings have been saved successfully.")

    def setup_settings_tab(self, parent):
        """
        Creates the UI for the settings tab, only adjusting the confidence thresholds
        of the OD and Bigface YOLO models in real-time.
        
        Args:
            parent: The parent frame (settings tab)
        """
        # Main container for settings
        settings_container = tk.Frame(parent, bg="#0a2158")
        settings_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Settings Title
        title_label = tk.Label(settings_container, text="Model Confidence Settings", 
                               font=("Arial", 18, "bold"), fg="white", bg="#0a2158")
        title_label.pack(pady=(0, 20))
        
        # ===== MODEL CONFIDENCE SECTION =====
        conf_frame = tk.LabelFrame(settings_container, text="Model Confidence Thresholds", 
                                   font=("Arial", 14, "bold"), fg="white", bg="#0a2158", bd=2)
        conf_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # OD Model Confidence Slider
        od_conf_frame = tk.Frame(conf_frame, bg="#0a2158", pady=10)
        od_conf_frame.pack(fill=tk.X, padx=10)
        
        od_conf_label = tk.Label(od_conf_frame, text="OD Model Confidence", font=("Arial", 12), 
                                 fg="white", bg="#0a2158", width=20, anchor="w")
        od_conf_label.pack(side=tk.LEFT, padx=10)
        
        # Initialize with default value (25% = 0.25)
        if not hasattr(self, 'od_conf_threshold'):
            self.od_conf_threshold = 0.25
        self.od_conf_slider_value = tk.DoubleVar(value=self.od_conf_threshold * 100)
        
        od_conf_slider = ttk.Scale(od_conf_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                   length=300, variable=self.od_conf_slider_value)
        od_conf_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.od_conf_value_label = tk.Label(od_conf_frame, text=f"{int(self.od_conf_threshold * 100)}%", 
                                            font=("Arial", 12), fg="white", bg="#0a2158", width=5)
        self.od_conf_value_label.pack(side=tk.LEFT, padx=10)
        
        # Update OD confidence value label when slider is moved
        def update_od_conf_label(val):
            self.od_conf_value_label.config(text=f"{int(float(val))}%")
            # Update the actual threshold value
            self.od_conf_threshold = float(val) / 100
            # If inspection is running, update the model in real-time
            if hasattr(self, 'inspection_running') and self.inspection_running:
                self.update_model_confidence()
        
        od_conf_slider.config(command=update_od_conf_label)
        
        # BIG FACE Model Confidence Slider
        bf_conf_frame = tk.Frame(conf_frame, bg="#0a2158", pady=10)
        bf_conf_frame.pack(fill=tk.X, padx=10)
        
        bf_conf_label = tk.Label(bf_conf_frame, text="Bigface Model Confidence", font=("Arial", 12), 
                                 fg="white", bg="#0a2158", width=20, anchor="w")
        bf_conf_label.pack(side=tk.LEFT, padx=10)
        
        # Initialize with default value (25% = 0.25)
        if not hasattr(self, 'bf_conf_threshold'):
            self.bf_conf_threshold = 0.25
        self.bf_conf_slider_value = tk.DoubleVar(value=self.bf_conf_threshold * 100)
        
        bf_conf_slider = ttk.Scale(bf_conf_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                   length=300, variable=self.bf_conf_slider_value)
        bf_conf_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        self.bf_conf_value_label = tk.Label(bf_conf_frame, text=f"{int(self.bf_conf_threshold * 100)}%", 
                                            font=("Arial", 12), fg="white", bg="#0a2158", width=5)
        self.bf_conf_value_label.pack(side=tk.LEFT, padx=10)
        
        # Update BF confidence value label when slider is moved
        def update_bf_conf_label(val):
            self.bf_conf_value_label.config(text=f"{int(float(val))}%")
            # Update the actual threshold value
            self.bf_conf_threshold = float(val) / 100
            # If inspection is running, update the model in real-time
            if hasattr(self, 'inspection_running') and self.inspection_running:
                self.update_model_confidence()
        
        bf_conf_slider.config(command=update_bf_conf_label)
        
        # Save button for settings
        save_button = tk.Button(settings_container, text="Save Settings", font=("Arial", 12, "bold"),
                                bg="#28a745", fg="white", command=self.save_thresholds)
        save_button.pack(pady=20)
    def setup_statistics_tab(self, parent):
        # Main statistics container
        stats_container = tk.Frame(parent, bg="#0a2158")
        stats_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(stats_container, text="Inspection Statistics", font=("Arial", 24, "bold"), 
                              fg="white", bg="#0a2158")
        title_label.pack(pady=(0, 30))
        
        # Total statistics
        total_stats_frame = tk.LabelFrame(stats_container, text="Total Statistics", font=("Arial", 16, "bold"), 
                                         fg="white", bg="#0a2158", bd=2)
        total_stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Total statistics variables
        self.total_inspected_var = tk.StringVar(value="0")
        self.total_defective_var = tk.StringVar(value="0")
        self.total_good_var = tk.StringVar(value="0")
        self.total_proportion_var = tk.StringVar(value="0%")
        
        total_stats_inner = tk.Frame(total_stats_frame, bg="#0a2158")
        total_stats_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create grid for total stats
        total_grid = tk.Frame(total_stats_inner, bg="#0a2158")
        total_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Total stats labels
        self.create_stat_label(total_grid, "Total Rollers Inspected:", self.total_inspected_var, 0)
        self.create_stat_label(total_grid, "Total Defective Rollers:", self.total_defective_var, 1)
        self.create_stat_label(total_grid, "Total Good Rollers:", self.total_good_var, 2)
        self.create_stat_label(total_grid, "Total Defective Proportion:", self.total_proportion_var, 3)
        
        # Create two frames for OD and BF statistics
        stats_frame = tk.Frame(stats_container, bg="#0a2158")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # OD Statistics
        od_stats_frame = tk.LabelFrame(stats_frame, text="OD Camera Statistics", font=("Arial", 16, "bold"), 
                                      fg="white", bg="#0a2158", bd=2)
        od_stats_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # OD Stats labels
        self.od_inspected_var = tk.StringVar(value="0")
        self.od_defective_var = tk.StringVar(value="0")
        self.od_good_var = tk.StringVar(value="0")
        self.od_proportion_var = tk.StringVar(value="0%")
        
        od_stats_inner = tk.Frame(od_stats_frame, bg="#0a2158")
        od_stats_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.create_stat_label(od_stats_inner, "Rollers Inspected:", self.od_inspected_var, 0)
        self.create_stat_label(od_stats_inner, "Defective Rollers:", self.od_defective_var, 1)
        self.create_stat_label(od_stats_inner, "Good Rollers:", self.od_good_var, 2)
        self.create_stat_label(od_stats_inner, "Defective Proportion:", self.od_proportion_var, 3)
        
        # OD Defect statistics
        od_defect_frame = tk.LabelFrame(od_stats_frame, text="Defect Types", font=("Arial", 14, "bold"), 
                                       fg="white", bg="#0a2158", bd=1)
        od_defect_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create grid for OD defect stats
        od_defect_grid = tk.Frame(od_defect_frame, bg="#0a2158")
        od_defect_grid.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # OD Defect headers
        od_headers = ["Defect Type", "Count", "Percentage"]
        for col, header in enumerate(od_headers):
            label = tk.Label(od_defect_grid, text=header, font=("Arial", 12, "bold"), 
                            fg="white", bg="#0a2158", padx=10, pady=5)
            label.grid(row=0, column=col, sticky="w")
        
        # OD Defect rows (mock data)
        for row, defect in enumerate(self.od_defect_thresholds.keys()):
            # Defect name
            label = tk.Label(od_defect_grid, text=defect, font=("Arial", 10), 
                            fg="white", bg="#0a2158", padx=10, pady=5, anchor="w")
            label.grid(row=row+1, column=0, sticky="w")
            
            # Count (mock data)
            count = np.random.randint(0, 50)
            label = tk.Label(od_defect_grid, text=str(count), font=("Arial", 10), 
                            fg="white", bg="#0a2158", padx=10, pady=5)
            label.grid(row=row+1, column=1)
            
            # Percentage (mock data)
            percentage = np.random.randint(1, 30)
            label = tk.Label(od_defect_grid, text=f"{percentage}%", font=("Arial", 10), 
                            fg="white", bg="#0a2158", padx=10, pady=5)
            label.grid(row=row+1, column=2)
        
        # BIG FACE Statistics
        bf_stats_frame = tk.LabelFrame(stats_frame, text="BIG FACE Camera Statistics", font=("Arial", 16, "bold"), 
                                      fg="white", bg="#0a2158", bd=2)
        bf_stats_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # BIG FACE Stats labels
        self.bf_inspected_var = tk.StringVar(value="0")
        self.bf_defective_var = tk.StringVar(value="0")
        self.bf_good_var = tk.StringVar(value="0")
        self.bf_proportion_var = tk.StringVar(value="0%")
        
        bf_stats_inner = tk.Frame(bf_stats_frame, bg="#0a2158")
        bf_stats_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.create_stat_label(bf_stats_inner, "Rollers Inspected:", self.bf_inspected_var, 0)
        self.create_stat_label(bf_stats_inner, "Defective Rollers:", self.bf_defective_var, 1)
        self.create_stat_label(bf_stats_inner, "Good Rollers:", self.bf_good_var, 2)
        self.create_stat_label(bf_stats_inner, "Defective Proportion:", self.bf_proportion_var, 3)
        
        # BIG FACE Defect statistics
        bf_defect_frame = tk.LabelFrame(bf_stats_frame, text="Defect Types", font=("Arial", 14, "bold"), 
                                       fg="white", bg="#0a2158", bd=1)
        bf_defect_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create grid for BF defect stats
        bf_defect_grid = tk.Frame(bf_defect_frame, bg="#0a2158")
        bf_defect_grid.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # BF Defect headers
        bf_headers = ["Defect Type", "Count", "Percentage"]
        for col, header in enumerate(bf_headers):
            label = tk.Label(bf_defect_grid, text=header, font=("Arial", 12, "bold"), 
                            fg="white", bg="#0a2158", padx=10, pady=5)
            label.grid(row=0, column=col, sticky="w")
        
        # BF Defect rows (mock data)
        for row, defect in enumerate(self.bf_defect_thresholds.keys()):
            # Defect name
            label = tk.Label(bf_defect_grid, text=defect, font=("Arial", 10), 
                            fg="white", bg="#0a2158", padx=10, pady=5, anchor="w")
            label.grid(row=row+1, column=0, sticky="w")
            
            # Count (mock data)
            count = np.random.randint(0, 50)
            label = tk.Label(bf_defect_grid, text=str(count), font=("Arial", 10), 
                            fg="white", bg="#0a2158", padx=10, pady=5)
            label.grid(row=row+1, column=1)
            
            # Percentage (mock data)
            percentage = np.random.randint(1, 30)
            label = tk.Label(bf_defect_grid, text=f"{percentage}%", font=("Arial", 10), 
                            fg="white", bg="#0a2158", padx=10, pady=5)
            label.grid(row=row+1, column=2)
        
        # Configure grid weights for stats frame
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)
        stats_frame.grid_rowconfigure(0, weight=1)
    
    def create_slider(self, parent, label_text, min_val, max_val, default_val, row, is_od=True):
        if not hasattr(self, "slider_values"):  
            self.slider_values = {}

        frame = tk.Frame(parent, bg="#0a2158")
        frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)

        label = tk.Label(frame, text=label_text, font=("Arial", 10), fg="white", bg="#0a2158", width=20, anchor="w")
        label.pack(side=tk.LEFT, padx=5)

        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=200)
        slider.set(default_val)  
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        value_label = tk.Label(frame, text=f"{default_val}%", font=("Arial", 10), fg="white", bg="#0a2158", width=5)
        value_label.pack(side=tk.RIGHT, padx=5)

        slider.configure(command=lambda val, lbl=value_label, defect=label_text, is_od=is_od: self.update_threshold(val, lbl, defect, is_od))

        self.slider_values[label_text] = slider  


    
    def create_stat_label(self, parent, label_text, var, row):
        frame = tk.Frame(parent, bg="#0a2158")
        frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        
        label = tk.Label(frame, text=label_text, font=("Arial", 10), fg="white", bg="#0a2158", width=20, anchor="w")
        label.pack(side=tk.LEFT, padx=5)
        
        value_label = tk.Label(frame, textvariable=var, font=("Arial", 10, "bold"), fg="white", bg="#0a2158", width=10)
        value_label.pack(side=tk.LEFT, padx=5)
    
    def start_camera_feeds(self):
        # For demonstration, we'll create mock camera feeds
        # In a real application, you would connect to actual cameras
        
        # Start threads for camera feeds
        self.camera_running = True
        
        self.od_thread = threading.Thread(target=self.update_od_camera)
        self.od_thread.daemon = True
        self.od_thread.start()
        
        self.bf_thread = threading.Thread(target=self.update_bf_camera)
        self.bf_thread.daemon = True
        self.bf_thread.start()
    
    def update_od_camera(self):

        while self.camera_running:
            with self.annotated_frame_lock_od:
                np_frame = np.frombuffer(self.shared_annotated_od.get_obj(), dtype=np.uint8).reshape(self.frame_shape)
                frame = np_frame.copy()

            resized_frame = cv2.resize(frame, (400, 250))
            # Convert frame to a format compatible with Tkinter
            img = PIL.Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            imgtk = PIL.ImageTk.PhotoImage(image=img)

            # Update the canvas
            self.od_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.od_canvas.image = imgtk

            time.sleep(0.03)  # Limit update rate

    def update_bf_camera(self):
        while self.camera_running:
            with self.annotated_frame_lock_bigface:
                np_frame = np.frombuffer(self.shared_annotated_bigface.get_obj(), dtype=np.uint8).reshape(self.frame_shape)
                frame = np_frame.copy()

            resized_frame = cv2.resize(frame, (400, 250))
            # Convert frame to a format compatible with Tkinter
            img = PIL.Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            imgtk = PIL.ImageTk.PhotoImage(image=img)

            # Update the canvas
            self.bf_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.bf_canvas.image = imgtk

            time.sleep(0.03)  # Limit update rate

    
    def update_statistics(self):
        # Simulate updating statistics
        if hasattr(self, 'od_inspected_var') and self.inspection_running:
            # Increment counters randomly for demonstration
            if np.random.random() < 0.2:  # 20% chance to update
                self.od_inspected += 1
                defect = np.random.random() < 0.3  # 30% chance of defect
                if defect:
                    self.od_defective += 1
                else:
                    self.od_good += 1
                
                # Update display variables
                self.od_inspected_var.set(str(self.od_inspected))
                self.od_defective_var.set(str(self.od_defective))
                self.od_good_var.set(str(self.od_good))
                
                if self.od_inspected > 0:
                    proportion = (self.od_defective / self.od_inspected) * 100
                    self.od_proportion_var.set(f"{proportion:.1f}%")
            
            # BIG FACE statistics
            if np.random.random() < 0.2:  # 20% chance to update
                self.bf_inspected += 1
                defect = np.random.random() < 0.2  # 20% chance of defect
                if defect:
                    self.bf_defective += 1
                else:
                    self.bf_good += 1
                
                # Update display variables
                self.bf_inspected_var.set(str(self.bf_inspected))
                self.bf_defective_var.set(str(self.bf_defective))
                self.bf_good_var.set(str(self.bf_good))
                
                if self.bf_inspected > 0:
                    proportion = (self.bf_defective / self.bf_inspected) * 100
                    self.bf_proportion_var.set(f"{proportion:.1f}%")
            
            # Update total statistics
            total_inspected = self.od_inspected + self.bf_inspected
            total_defective = self.od_defective + self.bf_defective
            total_good = self.od_good + self.bf_good
            
            self.total_inspected_var.set(str(total_inspected))
            self.total_defective_var.set(str(total_defective))
            self.total_good_var.set(str(total_good))
            
            if total_inspected > 0:
                total_proportion = (total_defective / total_inspected) * 100
                self.total_proportion_var.set(f"{total_proportion:.1f}%")
        
        # Schedule next update
        self.after(500, self.update_statistics)
    
    

    def initialize_system(self):

        self.RACK = 0
        self.PLC_IP = "172.17.8.17" 
        self.SLOT = 1
        self.DB_NUMBER = 86
        last_detection_time = 0
        DEBOUNCE_INTERVAL = 0.0
        initialize_bigface_csv()
        initialize_od_csv()

        print("Loading YOLO model...")
        self.model_bigface = YOLO(r"C:\Users\NBC\Desktop\DefectiveRollerDetection\OldModels\Bigfacelatest.pt")
        self.model_od = YOLO(r"C:\Users\NBC\Desktop\WELVISION-Project\FEBRUARY-13-ENDGAME\models\feb27.pt")

        self.model_bigface.to('cuda')
        self.model_od.to('cuda')



        self.frame_shape = (960, 1280, 3)

        self.manager = Manager()
        self.shared_data = self.manager.dict()
        self.shared_data['bigface'] = False
        self.shared_data['od'] = False
        self.shared_data['bigface_presence'] = False
        self.shared_data['od_presence'] = False

        self.command_queue = Queue()

        self.proximity_count_od = Value('i', 0)
        self.proximity_count_bigface = Value('i', 0)

        self.roller_data_od = self.manager.dict()
        self.roller_queue_od = Queue()
        self.roller_queue_bigface = Queue()
        self.roller_updation_dict = self.manager.dict()


        self.shared_frame_bigface = Array('B', np.zeros(self.frame_shape, dtype=np.uint8).flatten())
        self.shared_frame_od = Array('B', np.zeros(self.frame_shape, dtype=np.uint8).flatten())

        self.frame_lock_bigface = Lock()
        self.frame_lock_od = Lock()
        self.queue_lock = Lock()
    
        self.shared_annotated_bigface = Array('B', np.zeros(self.frame_shape, dtype=np.uint8).flatten())
        self.shared_annotated_od = Array('B', np.zeros(self.frame_shape, dtype=np.uint8).flatten())

        self.annotated_frame_lock_bigface = Lock()
        self.annotated_frame_lock_od = Lock()

    def on_closing(self):
        self.camera_running = False
        time.sleep(0.5)  
        self.destroy()

if __name__ == "__main__":

    app = WelVisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()