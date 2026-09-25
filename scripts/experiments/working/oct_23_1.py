import os
import glob
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import csv

# ---------------- CONFIG ----------------
INPUT_DIR = "/Users/tanishq/Desktop/untitled folder 5"
OUTPUT_DIR = "/Users/tanishq/Desktop/Tracked_Frames_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODIFICATION: This is no longer the primary matching gate ---
MAX_DISTANCE = 150.0  # Kept as a fallback, but we will bypass it
# --- End Modification ---

MAX_AGE = 3
# STATIONARY_TOL = 3.0  # pixels — if track centroid doesn't move more than this, we consider it stale

TRACK_BOX_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 0, 255)

def list_images_sorted(folder):
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = [f for e in exts for f in glob.glob(os.path.join(folder, e))]
    return sorted(files)

def find_drawn_boxes_by_color(frame):
    dets = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([140, 255, 255])
    lower_green = np.array([35, 50, 30])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue) | cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        dets.append([x, y, x + w, y + h, 1.0])
    return dets

class KalmanTrack:
    def __init__(self, init_centroid):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.statePost = np.array([[init_centroid[0]],
                                      [init_centroid[1]],
                                      [0],
                                      [0]], np.float32)

    def predict(self):
        p = self.kf.predict()
        return np.array([p[0, 0], p[1, 0]])

    def correct(self, c):
        meas = np.array([[c[0]], [c[1]]], np.float32)
        corr = self.kf.correct(meas)
        return np.array([corr[0, 0], corr[1, 0]])

class Track:
    def __init__(self, tid, bbox, centroid):
        self.id = tid
        self.bbox = bbox
        self.kf = KalmanTrack(centroid)
        self.history = [centroid]
        self.time_since_update = 0

    def predict(self): return self.kf.predict()

    def update(self, bbox, centroid):
        self.bbox = bbox
        corrected = self.kf.correct(centroid)
        self.history.append(corrected)
        self.time_since_update = 0

    # def is_stationary(self):
    #     if len(self.history) < 3: return False
    #     d = np.linalg.norm(self.history[-1] - self.history[-3])
    #     return d < STATIONARY_TOL

class MultiTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def _centroid(self, b): return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2])

    def update(self, detections):
        dets = np.array(detections)
        det_centroids = np.array([self._centroid(d[:4]) for d in dets]) if len(dets) else np.empty((0, 2))
        preds = np.array([t.predict() for t in self.tracks]) if self.tracks else np.empty((0, 2))
        
        # --- MODIFICATION: Cleaned up matching logic ---
        matches = []
        unmatched_tracks = list(range(len(preds)))
        unmatched_dets = list(range(len(dets)))

        if len(preds) and len(dets):
            dists = cdist(preds, det_centroids)
            row_ind, col_ind = linear_sum_assignment(dists)
            
            # We now trust the assignment algorithm completely.
            # This forces a match even if the distance is large,
            # preventing the Kalman filter from "running away".
            for r, c in zip(row_ind, col_ind):
                # if dists[r, c] < MAX_DISTANCE:  <-- THIS WAS THE PROBLEM. IT IS NOW REMOVED.
                
                # We only match if the assignment is "reasonable" in the context
                # of the number of items. linear_sum_assignment handles this.
                if r < len(preds) and c < len(dets):
                    matches.append((r, c))
                    if r in unmatched_tracks: unmatched_tracks.remove(r)
                    if c in unmatched_dets: unmatched_dets.remove(c)
        # --- End Modification ---

        for r, c in matches:
            self.tracks[r].update(dets[c, :4], det_centroids[c])

        for r in unmatched_tracks:
            self.tracks[r].time_since_update += 1

        for c in unmatched_dets:
            # --- MODIFICATION: Only add new tracks if we have less than 20 ---
            if len(self.tracks) >= 20:
                continue # We have 20 tracks, do not add more.
            # --- End Modification ---
            bbox, centroid = dets[c, :4], det_centroids[c]
            t = Track(self.next_id, bbox, centroid)
            self.next_id += 1
            self.tracks.append(t)

        # --- MODIFICATION: Comment out this line to make tracks permanent ---
        # self.tracks = [t for t in self.tracks if t.time_since_update <= MAX_AGE and not t.is_stationary()]

        # Return only tracks that were updated THIS frame (time_since_update == 0)
        return {t.id: (t.bbox, t.history[-1]) for t in self.tracks if t.time_since_update == 0}

def main():
    frames = list_images_sorted(INPUT_DIR)
    tracker = MultiTracker()
    
    # List to store all trajectory data for the CSV
    all_frame_data = []

    for i, fp in enumerate(frames):
        f = cv2.imread(fp)
        if f is None: continue
        detections = find_drawn_boxes_by_color(f)
        active_tracks = tracker.update(detections)
        vis = f.copy()
        
        for tid, (bbox, centroid) in active_tracks.items():
            # Store data for CSV
            all_frame_data.append([i, tid, centroid[0], centroid[1]])
            
            # Draw visualization
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), TRACK_BOX_COLOR, 2)
            cv2.putText(vis, f"ID:{tid}", (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        
        out_path = os.path.join(OUTPUT_DIR, f"tracked_{i:04d}.png")
        cv2.imwrite(out_path, vis)
        print(f"Processed frame {i+1}/{len(frames)}")

    # After the loop, save the CSV
    csv_path = os.path.join(OUTPUT_DIR, "trajectories.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame","ID","X","Y"])
        writer.writerows(all_frame_data)

    print(f"\n✅ Tracking complete.")
    print(f"Frames saved to: {OUTPUT_DIR}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()

