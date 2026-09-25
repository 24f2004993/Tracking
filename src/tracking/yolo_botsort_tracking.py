import os
import glob
import cv2
import numpy as np

# FIXED: This is the correct, specific import path for BoTSORT
from boxmot.trackers.botsort.bot_sort import BoTSORT
from ultralytics import YOLO

# --- Configuration ---
INPUT_DIR = "data/frames"
OUTPUT_DIR = "outputs/yolo_botsort"
MODEL_WEIGHTS = "models/osnet_x0_25_msmt17.pt" # This is the Re-ID model BoT-SORT uses
DEVICE = "cpu"

def run_tracker():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detector = YOLO('models/yolov8n.pt')

    # Initialize the BoT-SORT tracker
    tracker = BoTSORT(
        model_weights=MODEL_WEIGHTS,
        device=DEVICE,
        fp16=False
    )

    image_extensions = ("*.png", "*.jpg", "*.jpeg")
    frames = []
    for ext in image_extensions:
        frames.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    frames = sorted(frames)

    if not frames:
        print(f"No image files found in {INPUT_DIR}. Please check the path.")
        return

    print(f"Found {len(frames)} frames. Starting tracking with BoT-SORT...")

    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue

        results = detector(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()

        tracks = tracker.update(detections, frame)

        if tracks.shape[0] > 0:
            for t in tracks:
                x1, y1, x2, y2, track_id, cls, conf = t
                x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        out_path = os.path.join(OUTPUT_DIR, f"tracked_{i:04d}.png")
        cv2.imwrite(out_path, frame)
        print(f"Processed frame {i+1}/{len(frames)} -> Saved to {out_path}")

    print("--- Tracking complete! ---")

if __name__ == "__main__":
    run_tracker()
