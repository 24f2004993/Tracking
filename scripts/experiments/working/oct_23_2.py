import os
import glob
import cv2
import numpy as np
from collections import deque
import csv  # <-- 1. IMPORTED CSV MODULE

# ---------------- CONFIG ----------------
INPUT_DIR = '/Users/tanishq/Desktop/Tracked_Frames_Output'
OUTPUT_DIR = "/Users/tanishq/Desktop/particle3"

# --- 2. DEFINED CSV OUTPUT PATH ---
CSV_OUT = os.path.join(OUTPUT_DIR, "detected_centroids.csv")

MIN_GREEN_AREA = 20
SPLIT_MIN_AREA = 8

BOX_COLOR = (255, 0, 255)     # purple
TEXT_COLOR = (0, 0, 255)      # red
CENTROID_COLOR = (255, 0, 0)  # blue

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- utilities ----------------
def list_images_sorted(folder):
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = [f for e in exts for f in glob.glob(os.path.join(folder, e))]
    return sorted(files)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)

# ---------------- Detection splitting ----------------
def split_merged_green_regions(frame, green_mask):
    detections = []
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < MIN_GREEN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = green_mask[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, particle_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        particle_mask = cv2.bitwise_and(particle_mask, roi_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        particle_mask = cv2.morphologyEx(particle_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        inner_cnts, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ✅ If no inner contours — use center of mass
        if not inner_cnts:
            M = cv2.moments(roi_mask)
            if M["m00"] > 0:
                cx = x + int(M["m10"] / M["m00"])
                cy = y + int(M["m01"] / M["m00"])
                detections.append({'box': [x, y, x+w, y+h], 'id': f'p{i+1}', 'parent': i, 'center': (cx, cy)})
            else:
                detections.append({'box': [x, y, x+w, y+h], 'id': f'p{i+1}', 'parent': i})
            continue

        valid_inner = False
        for j, ic in enumerate(inner_cnts):
            if cv2.contourArea(ic) < SPLIT_MIN_AREA:
                continue
            valid_inner = True
            ix, iy, iw, ih = cv2.boundingRect(ic)
            cx = x + ix + iw // 2
            cy = y + iy + ih // 2
            detections.append({'box': [x+ix, y+iy, x+ix+iw, y+iy+ih],
                               'id': f'p{i+1}-{j+1}', 'parent': i, 'center': (cx, cy)})

        # If none of the inner contours qualify, add main centroid
        if not valid_inner:
            M = cv2.moments(roi_mask)
            if M["m00"] > 0:
                cx = x + int(M["m10"] / M["m00"])
                cy = y + int(M["m01"] / M["m00"])
                detections.append({'box': [x, y, x+w, y+h],
                                   'id': f'p{i+1}', 'parent': i, 'center': (cx, cy)})
    return detections

# ---------------- Main Loop ----------------
def run():
    frames = list_images_sorted(INPUT_DIR)

    # --- 3. WRITE CSV HEADER (before the loop) ---
    try:
        with open(CSV_OUT, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "ID", "X", "Y"])
    except IOError as e:
        print(f"Error: Could not write to CSV file. Check permissions. {e}")
        return
    # --- END ---

    for i, fp in enumerate(frames):
        frame = cv2.imread(fp)
        if frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k, iterations=1)

        detections = split_merged_green_regions(frame, green_mask)

        print(f"--- Frame {i:04d} ---")
        
        # --- 4. COLLECT ROWS FOR CSV ---
        csv_rows = []
        # --- END ---
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)

            # ✅ Draw centroid for every box
            if 'center' in det:
                cx, cy = det['center']
            else:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
            
            # Ensure cx, cy are floats for consistency
            cx, cy = float(cx), float(cy)

            cv2.circle(frame, (int(cx), int(cy)), 4, CENTROID_COLOR, -1)
            cv2.putText(frame, det['id'], (int(cx) + 8, int(cy) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

            print(f"{det['id']}: ({cx:.2f}, {cy:.2f})")
            
            # --- 5. ADD DATA TO LIST ---
            csv_rows.append([i, det['id'], cx, cy])
            # --- END ---

        # --- 6. APPEND ROWS TO CSV (after processing all detections for this frame) ---
        if csv_rows:
            try:
                with open(CSV_OUT, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(csv_rows)
            except IOError as e:
                print(f"Warning: Could not append to CSV file. {e}")
        # --- END ---

        out_path = os.path.join(OUTPUT_DIR, f"coords_{i:04d}.png")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}\n")

if __name__ == "__main__":
    run()
