import os
import glob
import cv2
import numpy as np
from collections import deque

# ---------------- CONFIG ----------------
INPUT_DIR = "/Users/tanishq/Desktop/untitled folder 2"
OUTPUT_DIR = "/Users/tanishq/Desktop/oct6_merged_coords_outpu2"

MIN_GREEN_AREA = 20
SPLIT_MIN_AREA = 8

BOX_COLOR = (255, 0, 255)   # purple
TEXT_COLOR = (0, 0, 255)    # red
CENTROID_COLOR = (255, 0, 0) # blue

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- utilities ----------------
def list_images_sorted(folder):
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = [f for e in exts for f in glob.glob(os.path.join(folder, e))]
    return sorted(files)

def iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)

# ---------------- Detection splitting (Your function) ----------------
def split_merged_green_regions(frame, green_mask):
    detections = []
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < MIN_GREEN_AREA: continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = green_mask[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, particle_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        particle_mask = cv2.bitwise_and(particle_mask, roi_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        particle_mask = cv2.morphologyEx(particle_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        inner_cnts, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not inner_cnts:
            detections.append({'box': [x, y, x+w, y+h], 'id': f'p{i+1}'})
            continue
        for j, ic in enumerate(inner_cnts):
            if cv2.contourArea(ic) < SPLIT_MIN_AREA: continue
            ix, iy, iw, ih = cv2.boundingRect(ic)
            detections.append({'box': [x+ix, y+iy, x+ix+iw, y+iy+ih], 'id': f'p{i+1}-{j+1}'})
    return detections

# ---------------- NEW: Collision Processing Function ----------------
def get_final_coordinates(detections):
    if not detections:
        return {}

    # Build adjacency list for collision graph
    adj = {i: [] for i in range(len(detections))}
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if iou(detections[i]['box'], detections[j]['box']) > 0:
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components (collision groups)
    visited = set()
    groups = []
    for i in range(len(detections)):
        if i not in visited:
            group = []
            q = deque([i])
            visited.add(i)
            while q:
                u = q.popleft()
                group.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            groups.append(group)

    # Calculate final coordinates
    final_coords = {}
    for group in groups:
        boxes = [detections[i]['box'] for i in group]
        if len(group) == 1:
            # Not a collision, use its own centroid
            box = boxes[0]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            final_coords[detections[group[0]]['id']] = (cx, cy)
        else:
            # Collision group: calculate a single group centroid
            min_x = min(b[0] for b in boxes)
            min_y = min(b[1] for b in boxes)
            max_x = max(b[2] for b in boxes)
            max_y = max(b[3] for b in boxes)
            group_cx = (min_x + max_x) / 2.0
            group_cy = (min_y + max_y) / 2.0
            # Assign the same coordinate to all particles in the group
            for i in group:
                final_coords[detections[i]['id']] = (group_cx, group_cy)
    return final_coords

# ---------------- Main Loop ----------------
def run():
    frames = list_images_sorted(INPUT_DIR)

    for i, fp in enumerate(frames):
        frame = cv2.imread(fp)
        if frame is None: continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50]); upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k, iterations=1)

        # 1. Get precise detections using YOUR smart function
        detections = split_merged_green_regions(frame, green_mask)
        
        # 2. Process collisions to get final coordinates
        final_coordinates = get_final_coordinates(detections)

        # 3. Visualize and Print Results
        print(f"--- Frame {i:04d} ---")
        for p_id, coords in final_coordinates.items():
            print(f"Particle {p_id}: ({coords[0]:.2f}, {coords[1]:.2f})")
            # Draw the final coordinate as a blue dot on the image
            cv2.circle(frame, (int(coords[0]), int(coords[1])), 5, CENTROID_COLOR, -1)
            cv2.putText(frame, p_id, (int(coords[0]) + 8, int(coords[1]) + 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)
        
        # Draw the original detection boxes for context
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)

        out_path = os.path.join(OUTPUT_DIR, f"coords_{i:04d}.png")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}\n")

if __name__ == "__main__":
    run()