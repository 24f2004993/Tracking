import os
import glob
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# ---------------- CONFIG ----------------
INPUT_DIR = 'data/frames'
OUTPUT_DIR = "outputs/kalman_tracking"

MIN_GREEN_AREA = 20
SPLIT_MIN_AREA = 8
MAX_DISTANCE = 30.0
IOU_MIN = 0.1

BOX_COLOR = (255, 0, 255)   # purple
TEXT_COLOR = (0, 0, 255)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- utilities ----------------
def list_images_sorted(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = [f for e in exts for f in glob.glob(os.path.join(folder, e))]
    return sorted(files)


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    w = max(0.0, x2 - x1); h = max(0.0, y2 - y1)
    inter = w * h
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter + 1e-6
    return inter / union if union > 0 else 0


# ---------------- Detection splitting ----------------
def split_merged_green_regions(frame, green_mask):
    detections = []
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_GREEN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = green_mask[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        _, particle_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        particle_mask = cv2.bitwise_and(particle_mask, roi_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        particle_mask = cv2.morphologyEx(particle_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        inner_cnts, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(inner_cnts) == 0:
            detections.append([x, y, x+w, y+h, 1.0])
            continue
        for ic in inner_cnts:
            a = cv2.contourArea(ic)
            if a < SPLIT_MIN_AREA:
                continue
            ix, iy, iw, ih = cv2.boundingRect(ic)
            bx1 = x + ix; by1 = y + iy; bx2 = bx1 + iw; by2 = by1 + ih
            detections.append([bx1, by1, bx2, by2, 1.0])
    return detections


# ---------------- Tracker with Kalman ----------------
class KalmanTrack:
    def __init__(self, init_centroid):
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],
                                             [0,1,0,1],
                                             [0,0,1,0],
                                             [0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.statePre = np.array([[init_centroid[0]],
                                     [init_centroid[1]],
                                     [0],
                                     [0]], np.float32)
        self.kf.statePost = np.array([[init_centroid[0]],
                                      [init_centroid[1]],
                                      [0],
                                      [0]], np.float32)

    def predict(self):
        pred = self.kf.predict()
        return np.array([pred[0,0], pred[1,0]])

    def correct(self, centroid):
        meas = np.array([[centroid[0]], [centroid[1]]], np.float32)
        corrected = self.kf.correct(meas)
        return np.array([corrected[0,0], corrected[1,0]])


class DetectionKalmanTracker:
    def __init__(self, max_distance=MAX_DISTANCE, iou_min=IOU_MIN):
        self.max_distance = max_distance
        self.iou_min = iou_min
        self.next_id = 1
        self.active_tracks = {}  # id -> (bbox, centroid, KalmanTrack)

    def update(self, prev_tracks, detections):
        dets = np.array(detections) if len(detections) > 0 else np.empty((0,5))
        det_centroids = np.array([[ (d[0]+d[2])/2.0, (d[1]+d[3])/2.0 ] for d in dets]) \
                        if dets.shape[0] > 0 else np.empty((0,2))

        prev_ids = list(prev_tracks.keys())
        prev_boxes = [prev_tracks[i][0] for i in prev_ids]
        prev_centroids = np.array([prev_tracks[i][1] for i in prev_ids]) \
                         if len(prev_ids) > 0 else np.empty((0,2))
        prev_kf_objs = {tid: prev_tracks[tid][2] for tid in prev_ids}

        # Predict step for Kalman
        for tid in prev_ids:
            prev_centroids[prev_ids.index(tid)] = prev_kf_objs[tid].predict()

        matches = []
        unmatched_prev, unmatched_det = set(range(len(prev_ids))), set(range(len(dets)))

        if len(prev_ids) > 0 and len(dets) > 0:
            dists = cdist(prev_centroids, det_centroids)
            for r in range(dists.shape[0]):
                for c in range(dists.shape[1]):
                    if dists[r,c] > self.max_distance:
                        dists[r,c] = 1e6
            row_ind, col_ind = linear_sum_assignment(dists)
            for r, c in zip(row_ind, col_ind):
                if dists[r,c] < 1e6 and iou(prev_boxes[r], dets[c,:4]) >= self.iou_min:
                    matches.append((r,c))
                    unmatched_prev.discard(r); unmatched_det.discard(c)

        active_tracks = {}
        for r, c in matches:
            tid = prev_ids[r]
            bbox = dets[c,:4]
            centroid = det_centroids[c]
            kf_obj = prev_kf_objs[tid]
            corrected_centroid = kf_obj.correct(centroid)
            active_tracks[tid] = (bbox, corrected_centroid, kf_obj)

        for c in unmatched_det:
            tid = self.next_id; self.next_id += 1
            bbox = dets[c,:4]; centroid = det_centroids[c]
            kf_obj = KalmanTrack(centroid)
            active_tracks[tid] = (bbox, centroid, kf_obj)

        self.active_tracks = active_tracks
        return active_tracks


# ---------------- Main Loop ----------------
def run_tracker():
    frames = list_images_sorted(INPUT_DIR)
    tracker = DetectionKalmanTracker()
    prev_tracks = {}

    for i, fp in enumerate(frames):
        frame = cv2.imread(fp)
        if frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 80, 80]); upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k, iterations=1)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k, iterations=1)

        detections = split_merged_green_regions(frame, green_mask)
        tracks = tracker.update(prev_tracks, detections)

        # Draw bounding boxes only
        for tid, (bbox, centroid, _) in tracks.items():
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2), BOX_COLOR, 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)

        prev_tracks = tracks.copy()

        out_path = os.path.join(OUTPUT_DIR, f"tracked_{i:04d}.png")
        cv2.imwrite(out_path, frame)
        print("Saved", out_path)


if __name__ == "__main__":
    run_tracker()
