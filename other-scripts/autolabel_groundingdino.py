import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" 

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = "frame_6881_png.rf.4327cc308e9e0d709b556c6b7c5c2fc5.jpg"
image = Image.open(image_path)

image = image.convert("RGB")

text_labels = [["circular object"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

with open("detected_coordinates.txt", "w") as f:
    result = results[0]
    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        f.write(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}\n")

draw = ImageDraw.Draw(image)
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), f"{round(score.item(), 3)}", fill="red")

image.save("image_with_bboxes.png")

print("Coordinates saved to 'detected_coordinates.txt' and image saved as 'image_with_bboxes.png'")