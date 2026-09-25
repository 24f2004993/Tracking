import cv2
import os

# Folder containing images
input_folder = "path/to/project/predict14"
output_folder = "path/to/project/output14"
# Function to extract YOLO-style boxes from one image
def extract_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = image.shape[:2]
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Convert to YOLO format (normalized cx, cy, w, h)
        cx = (x + w / 2) / w_img
        cy = (y + h / 2) / h_img
        nw = w / w_img
        nh = h / h_img
        boxes.append((0, round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)))  # class 0

    return boxes

# Loop over all images in folder
os.makedirs(output_folder, exist_ok=True)
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Failed to read image: {filename}")
            continue

        boxes = extract_boxes(image)

        # Save YOLO label file
        label_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
        with open(label_path, "w") as f:
            for box in boxes:
                f.write(f"{box[0]} {' '.join(map(str, box[1:]))}\n")

        print(f"✅ Labels saved: {os.path.basename(label_path)}")

print("\n🎯 Done! All YOLO labels saved in the same folder.")