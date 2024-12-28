import cv2
import os
from ultralytics import YOLO
import time  # To measure the time taken for each image

# Load your custom-trained YOLOv8 model
model = YOLO(r'C:\Users\NBC\Desktop\DefectiveRollerDetection\OldModels\ODlatestmodel.pt')  # Path to your custom-trained model

# Check if the model is using GPU (if CUDA is available)
device = 'cuda' if model.device.type == 'cuda' else 'cpu'
print(f"Using device: {device}")

# Input and output directories
input_dir = r'C:\Users\NBC\Videos\frames_2'  # Path to the folder containing images
output_dir = r'C:\Users\NBC\Desktop\DefectiveRollerDetection\Output_FRAMES2'  # Path to save output images with detections

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all image files from the input directory (you can modify the file extensions if needed)
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# Process each image
for image_file in image_files:
    # Read the image
    image_path = os.path.join(input_dir, image_file)
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image {image_file}")
        continue
    
    # Start the timer to measure how long each image takes to process
    start_time = time.time()

    # Perform inference on the current image
    results = model(frame)  # 'frame' is the current image
    
    # Extract results (boxes, confidences, and class labels)
    for result in results:
        boxes = result.boxes.xyxy  # Get bounding box coordinates
        confidences = result.boxes.conf  # Get confidence scores
        class_ids = result.boxes.cls  # Get class ids
        labels = result.names  # Get class labels

        # Loop through the detections and draw bounding boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]  # Get the box coordinates
            confidence = confidences[i]  # Confidence score
            class_id = int(class_ids[i])  # Class ID
            label = labels[class_id]  # Class label

            # Draw bounding box and label on the image
            if confidence > 0.3:  # Only draw boxes with a confidence score above 50%
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green bounding box
                cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the annotated image to the output directory
    output_image_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_image_path, frame)

    # Display the output frame with a small delay to ensure the window doesn't freeze
    cv2.imshow("Detection Result", frame)
    
    # Wait for a short period (e.g., 1 millisecond) or press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print the time taken for processing this image
    print(f"Processed and saved: {output_image_path}. Time taken: {time.time() - start_time:.2f} seconds")

# Close all OpenCV windows after processing all images
cv2.destroyAllWindows()

print("All images processed successfully!")
