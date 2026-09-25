import cv2
import os
import numpy as np

# Update this path
image_folder = "path/to/project/predict14"
output_folder = os.path.join(image_folder, "yolo_labels")
os.makedirs(output_folder, exist_ok=True)

# Define color ranges in HSV for blue and green
color_ranges = {
    0: {"lower": (90, 100, 100), "upper": (130, 255, 255)},  # Blue
    1: {"lower": (40, 40, 40), "upper": (90, 255, 255)}       # Green
}

def extract_boxes(image, label_class):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(color_ranges[label_class]["lower"]),
                             np.array(color_ranges[label_class]["upper"]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image.shape[:2]
    boxes = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        width = bw / w
        height = bh / h
        boxes.append((label_class, x_center, y_center, width, height))

    return boxes

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        all_boxes = []
        for label_class in color_ranges:
            boxes = extract_boxes(img, label_class)
            all_boxes.extend(boxes)

        # Save YOLO annotations
        label_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
        with open(label_path, "w") as f:
            for box in all_boxes:
                f.write(f"{box[0]} {' '.join(map(str, box[1:]))}\n")

print(f"✅ YOLO annotations saved to: {output_folder}")

