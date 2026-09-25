import cv2
import pytesseract
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Path setup
frames_dir = '/Users/tanishq/Desktop/untitled folder 7'
output_dir = "/Users/tanishq/Desktop/particle_tracking_output21"
os.makedirs(output_dir, exist_ok=True)

# CSV output path
csv_path = os.path.join(output_dir, "particle_trajectories.csv")

# Tesseract setup (modify path if needed)
# For Mac:
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# Distance threshold for collision detection (in pixels)
COLLISION_THRESHOLD = 25

# Function to extract particle data from frame
def extract_particle_data(image, frame_id):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = []
    
    # Detect bounding boxes by color (blue/pink boxes assumed visible)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 100, 50), (140, 255, 255))  # blue range (adjust if needed)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # OCR to read IDs
    ocr_text = pytesseract.image_to_string(gray)
    ids = re.findall(r'\b\d+\b', ocr_text)
    ids = [int(i) for i in ids]
    
    id_index = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        if id_index < len(ids):
            pid = ids[id_index]
            id_index += 1
        else:
            pid = -1  # Unknown ID
        data.append((frame_id, pid, cx, cy))
    return data

# Process frames
all_data = []
frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])

for idx, frame_file in enumerate(frame_files):
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue
    frame_data = extract_particle_data(frame, idx + 1)
    all_data.extend(frame_data)
    print(f"Processed frame {idx + 1}/{len(frame_files)}")

# Create DataFrame
df = pd.DataFrame(all_data, columns=["frame_id", "particle_id", "x", "y"])

# Detect collisions
df["collided_with"] = ""
for i in range(len(df)):
    same_frame = df[df["frame_id"] == df.loc[i, "frame_id"]]
    x1, y1 = df.loc[i, "x"], df.loc[i, "y"]
    pid = df.loc[i, "particle_id"]
    collisions = []
    for j in range(len(same_frame)):
        if i == j:
            continue
        x2, y2 = same_frame.iloc[j]["x"], same_frame.iloc[j]["y"]
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist < COLLISION_THRESHOLD:
            collisions.append(str(same_frame.iloc[j]["particle_id"]))
    df.at[i, "collided_with"] = ",".join(collisions)

# Save CSV
df.to_csv(csv_path, index=False)
print(f"\nTrajectory data saved to: {csv_path}")

# Generate trajectory map
plt.figure(figsize=(10, 10))
unique_ids = df["particle_id"].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))

for pid, color in zip(unique_ids, colors):
    sub = df[df["particle_id"] == pid]
    if len(sub) > 1:
        plt.plot(sub["x"], sub["y"], marker="o", color=color, label=f"ID {pid}")

plt.title("Particle Trajectories Map")
plt.xlabel("X position (px)")
plt.ylabel("Y position (px)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()

map_path = os.path.join(output_dir, "trajectory_map.jpg")
plt.savefig(map_path, dpi=300)
plt.close()

print(f"Trajectory map saved to: {map_path}")
