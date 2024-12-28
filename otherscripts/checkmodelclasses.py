from ultralytics import YOLO

# Load the model
model = YOLO(r'C:\Users\NBC\Desktop\DefectiveRollerDetection\NewModels\bigfacev8.pt')  # Ensure this path is correct and the model exists

# Print the class names
print(model.names)
