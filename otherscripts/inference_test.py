import time
import cv2
import torch
from ultralytics import YOLO

def run_inference_on_video(model, source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    total_inference_time = 0
    min_inference_time = float('inf')
    max_inference_time = 0

    video_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        start_time = time.time()

        results = model.predict(frame, device=0)

        end_time = time.time()

        inference_time = end_time - start_time
        total_inference_time += inference_time
        min_inference_time = min(min_inference_time, inference_time)
        max_inference_time = max(max_inference_time, inference_time)
        frame_count += 1

        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        print(f"Frame {frame_count}: Inference Time: {inference_time:.4f} seconds")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0

    video_end_time = time.time()
    video_processing_time = video_end_time - video_start_time

    fps = frame_count / video_processing_time if video_processing_time > 0 else 0

    print("\nOverall Inference Stats:")
    print(f"Min Inference Time: {min_inference_time:.4f} seconds")
    print(f"Max Inference Time: {max_inference_time:.4f} seconds")
    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
    print(f"FPS Achieved: {fps:.2f} frames per second")
    print(f"Total Frames Processed: {frame_count}")


def main():
    model = YOLO("FEB27MODEL.pt")  

    if torch.cuda.is_available():
        model.to('cuda')
        print("Model loaded to GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    source = "damageonse.avi"  

    run_inference_on_video(model, source)

if __name__ == "__main__":
    main()
