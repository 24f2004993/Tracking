import os
import glob
import argparse
import cv2
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# ---------- IoU and bbox conversions ----------

def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two sets of boxes.
    - bb_test: Detections, shape [N, 4] (x1,y1,x2,y2)
    - bb_gt: Track predictions, shape [M, 4] (x1,y1,x2,y2)
    Returns: IoU matrix of shape [N, M]
    """
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]))

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt   = (bb_gt[..., 2]   - bb_gt[..., 0])   * (bb_gt[..., 3]   - bb_gt[..., 1])

    union = area_test + area_gt - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou

def convert_bbox_to_z(bbox):
    """
    bbox [x1, y1, x2, y2] -> z = [x, y, s, r]^T
    x,y is centre, s is scale (area), r is aspect ratio w/h
    """
    x1, y1, x2, y2 = bbox[:4]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    x = [x, y, s, r, x', y', s']^T -> [x1, y1, x2, y2] (optionally + score)
    """
    x_c, y_c, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    w = np.sqrt(max(1e-6, s * r))
    h = max(1e-6, s) / w
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

# ---------- KalmanBoxTracker (SORT) ----------

class KalmanBoxTracker(object):
    """
    Tracks an individual object with a Kalman filter.
    State: [x, y, s, r, x', y', s']^T
    Measure: [x, y, s, r]^T
    """
    count = 0

    def __init__(self, bbox):
        # 7D state, 4D measurement
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition F
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Measurement matrix H
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0],  # s
            [0, 0, 0, 1, 0, 0, 0],  # r
        ], dtype=float)

        # Covariances
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # high uncertainty for initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = deque(maxlen=30)  # store center points for trails
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """Update with observed bbox."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """Advance state and return predicted bbox."""
        # keep scale positive
        if (self.kf.x[2] + self.kf.x[6]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # save center for trail
        cx, cy = float(self.kf.x[0]), float(self.kf.x[1])
        self.history.append((int(round(cx)), int(round(cy))))

        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

# ---------- SORT multi-object tracker ----------

class Sort(object):
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        """
        max_age: max frames to keep track without updates
        min_hits: min hits before outputting a track
        iou_threshold: min IoU for a match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        dets: [[x1,y1,x2,y2,score], ...]
        Returns: [[x1,y1,x2,y2,id], ...]
        """
        self.frame_count += 1

        # predict new locations for existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()  # 1x4
            trks[t, :] = pos[0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)

        # if we have detections and trackers, associate
        if dets.size > 0 and trks.size > 0:
            iou_matrix = iou_batch(dets[:, :4], trks)
            # Hungarian on cost = 1 - IoU (maximize IoU)
            cost_matrix = 1.0 - iou_matrix
            rows, cols = linear_sum_assignment(cost_matrix)

            matched_indices = np.stack([rows, cols], axis=1)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d in range(dets.shape[0]):
            if d not in matched_indices[:, 0] if matched_indices.size else True:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(trks.shape[0]):
            if t not in matched_indices[:, 1] if matched_indices.size else True:
                unmatched_trackers.append(t)

        # filter matches with low IoU
        matches = []
        for m in matched_indices:
            d_idx, t_idx = m[0], m[1]
            if dets.size > 0 and trks.size > 0:
                iou_val = iou_batch(dets[d_idx:d_idx+1, :4], trks[t_idx:t_idx+1, :])[0, 0]
            else:
                iou_val = 0.0
            if iou_val < self.iou_threshold:
                unmatched_detections.append(d_idx)
                unmatched_trackers.append(t_idx)
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        # update matched trackers with assigned detections
        for m in matches:
            t_idx = m[1]
            d_idx = m[0]
            self.trackers[t_idx].update(dets[d_idx, :4])

        # create new trackers for unmatched detections
        for i in unmatched_detections:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # output tracks
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]  # [x1,y1,x2,y2]
            # only output active tracks
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            # remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

# ---------- Utilities ----------

def list_images_sorted(folder):
    """Returns a sorted list of image files from a directory."""
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)

# ---------- Main ----------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Detector: MOG2 (generic motion)
    detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # SORT tracker
    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_thresh)

    # colors per ID
    track_colors = {}

    frames = list_images_sorted(args.input_dir)
    if not frames:
        print(f"No images found in {args.input_dir}. Please check the path.")
        return

    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # --- Detection phase (simple foreground seg) ---
        fg_mask = detector.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < args.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, x + w, y + h, 1.0])

        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array(detections, dtype=float)

        # --- Tracking phase ---
        tracked = mot_tracker.update(dets)

        # --- Visualization ---
        for obj in tracked:
            x1, y1, x2, y2, obj_id = obj.astype(int)

            if obj_id not in track_colors:
                track_colors[obj_id] = (
                    int(np.random.randint(0, 255)),
                    int(np.random.randint(0, 255)),
                    int(np.random.randint(0, 255)),
                )
            color = track_colors[obj_id]

            # draw bbox and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"ID: {obj_id}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )

            # draw trajectory trail using the corresponding tracker history
            for trk in mot_tracker.trackers:
                if trk.id + 1 == obj_id:
                    if len(trk.history) > 1:
                        for j in range(1, len(trk.history)):
                            p0 = trk.history[j - 1]
                            p1 = trk.history[j]
                            if p0 is None or p1 is None:
                                continue
                            cv2.line(frame, p0, p1, color, 2)
                    break

        out_path = os.path.join(args.output_dir, f"track_{i:04d}.png")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SORT Object Tracker")
    parser.add_argument(
        '--input_dir', type=str,
        default="data/continuous_collision_frames",
        help='Path to the input image sequence directory.'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default="outputs/green_box_tracking",
        help='Path to the output directory.'
    )
    parser.add_argument('--min_area', type=int, default=200, help='Minimum contour area to be a detection.')
    parser.add_argument('--max_age', type=int, default=5, help='Max frames to keep a track without updates.')
    parser.add_argument('--min_hits', type=int, default=2, help='Min hits to start a track.')
    parser.add_argument('--iou_thresh', type=float, default=0.2, help='IOU threshold for matching.')

    args = parser.parse_args()
    main(args)
