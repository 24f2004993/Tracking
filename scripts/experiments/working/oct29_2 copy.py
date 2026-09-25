import cv2
import numpy as np
import os

# Input folder (same as before)
input_folder = '/Users/tanishq/Desktop/untitled folder 8'

# Define HSV range for bright green boxes
lower_green = np.array([35, 80, 80])
upper_green = np.array([90, 255, 255])

# Go through frames in order
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    frame_path = os.path.join(input_folder, filename)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up noise (helps when green boxes are faint)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)

    # Find contours of green boxes (if any)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red dots wherever green boxes appear
    if contours:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 40:  # skip small artifacts
                continue
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # Overwrite the same frame
    cv2.imwrite(frame_path, frame)

print("✅ All frames updated — red dots added at green box centers (collisions).")
