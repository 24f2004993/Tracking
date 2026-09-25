#!/usr/bin/env python3
"""
nov15_1_fixed_v4.py
Improved donut tracker — stronger donut filtering + stricter assignment to prevent ID swaps.
Produces:
 - tracked_particles.csv
 - trajectory_map.png
 - frames_debug/frame_####.png (visual debug)
"""

import os, glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# ---------------- USER PARAMETERS (tweak these if needed) ----------------
FRAMES_GLOB = "/Users/tanishq/Desktop/New Folder With Items/*.jpg"
OUTPUT_DIR = "/Users/tanishq/Desktop/particle_tracking_output_masked"
EXPECTED_TRACKS = 20

R_MIN, R_MAX = 5, 20
MERGE_DIST = 6

# stronger gating and penalty to reduce ID switching
MAX_DISP = 12              # px maximum allowed displacement per frame (tighter)
MOTION_WEIGHT = 0.65       # weight assigned to motion score
APPEARANCE_WEIGHT = 0.25   # appearance still used but lower
SWITCH_PENALTY = 0.8       # large penalty for switching identity (0..1)
TEMPLATE_SIZE = 31
TEMPLATE_UPDATE_ALPHA = 0.12
COLLISION_DIST = 12        # cluster distance to declare collision/occlusion
MAX_MISSED = 8
SAVE_DEBUG_FRAMES = True
# ------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_DEBUG_FRAMES:
    os.makedirs(os.path.join(OUTPUT_DIR, "frames_debug"), exist_ok=True)


def load_frames(glob_path):
    files = sorted(glob.glob(glob_path))
    if not files:
        raise FileNotFoundError("No frames matched: " + glob_path)
    imgs = [cv2.imread(f, cv2.IMREAD_COLOR) for f in files]
    return files, imgs


def ensure_gray(img):
    if img is None:
        raise ValueError("None image")
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return img.copy()


def make_circle_mask(shape, center, radius):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
    return mask


def merge_close(points, dist):
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    pts = np.array(points, dtype=float)
    used = np.zeros(len(pts), bool)
    out = []
    for i in range(len(pts)):
        if used[i]:
            continue
        group = [pts[i]]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if not used[j] and np.linalg.norm(pts[i] - pts[j]) <= dist:
                used[j] = True
                group.append(pts[j])
        out.append(np.mean(group, axis=0))
    return np.array(out)


# --- donut quality tests --------------------------------------------------
def is_good_donut(gray, x, y, r_est):
    """
    Quick donut test:
     - r_est: estimated radius (from contour)
     - check ring contrast: mean intensity in inner disk vs annulus
     - check circularity
    """
    h, w = gray.shape
    r = max(3, int(round(r_est)))
    if r < 3:
        return False
    x0 = int(round(x)); y0 = int(round(y))
    if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
        return False

    # sample patch bounded by 3*r
    sz = min(max(2 * r + 1, 21), 2 * r + 1 + 20)
    half = sz // 2
    x1 = max(0, x0 - half); x2 = min(w, x0 + half + 1)
    y1 = max(0, y0 - half); y2 = min(h, y0 + half + 1)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return False

    # compute radial mask
    yy, xx = np.mgrid[y1:y2, x1:x2]
    rr = np.hypot(xx - x0, yy - y0)
    inner_mask = rr <= max(1, 0.4 * r)
    ann_mask = (rr >= 0.6 * r) & (rr <= 1.1 * r)
    if ann_mask.sum() < 8 or inner_mask.sum() < 4:
        return False

    inner_mean = float(patch[inner_mask].mean())
    ann_mean = float(patch[ann_mask].mean())
    # donut interior should be darker than rim (since donuts are black)
    contrast = ann_mean - inner_mean
    if contrast < 10:   # threshold - tune if needed
        return False
    return True
# -------------------------------------------------------------------------


def detect_donuts(gray, mask):
    """Contour-based donut detection with Hough fallback, returning Nx2 centers."""
    proc = cv2.GaussianBlur(gray, (3, 3), 0)
    proc = cv2.equalizeHist(proc)

    # binary: highlight dark holes
    _, th = cv2.threshold(proc, 200, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(th)

    # ensure mask dtype & size match
    mask_u = mask
    if mask_u.dtype != np.uint8:
        mask_u = (mask_u > 0).astype(np.uint8) * 255
    if mask_u.shape != inv.shape:
        mask_u = cv2.resize(mask_u, (inv.shape[1], inv.shape[0]))
    inv = cv2.bitwise_and(inv, inv, mask=mask_u)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8:     # reject tiny specks
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < R_MIN - 1 or r > R_MAX + 2:
            continue
        # quick circularity test
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ < 0.2:   # looser circularity guard
            continue

        # ring contrast test
        if not is_good_donut(proc, x, y, r):
            continue

        # inside main imaging circle
        dx = x - CIRCLE_CENTER[0]; dy = y - CIRCLE_CENTER[1]
        if dx * dx + dy * dy > CIRCLE_RADIUS * CIRCLE_RADIUS:
            continue
        centers.append((x, y, r))

    # keep centers (x,y) only
    centers_xy = [(c[0], c[1]) for c in centers]
    centers_xy = merge_close(centers_xy, MERGE_DIST)

    # Hough fallback only if we have too few
    if centers_xy.shape[0] < EXPECTED_TRACKS // 2:
        circles = cv2.HoughCircles(proc, cv2.HOUGH_GRADIENT,
                                   dp=1.15, minDist=8,
                                   param1=60, param2=12,
                                   minRadius=max(1, R_MIN - 2), maxRadius=R_MAX + 2)
        if circles is not None:
            cur = np.squeeze(circles)
            if cur.ndim == 1:
                cur = cur[np.newaxis, :]
            for c in cur:
                x, y, r = float(c[0]), float(c[1]), float(c[2])
                dx = x - CIRCLE_CENTER[0]; dy = y - CIRCLE_CENTER[1]
                if dx * dx + dy * dy <= CIRCLE_RADIUS * CIRCLE_RADIUS:
                    if is_good_donut(proc, x, y, r):
                        centers_xy = np.vstack([centers_xy, (x, y)]) if centers_xy.size else np.array([(x, y)])
            centers_xy = merge_close(centers_xy, MERGE_DIST)

    if centers_xy.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.array(centers_xy, dtype=float)


def extract_ring_template(gray, x, y, size):
    """Normalized ring patch used for appearance matching."""
    h, w = gray.shape
    r = size // 2
    x0 = int(round(x)) - r
    y0 = int(round(y)) - r
    x1 = x0 + size
    y1 = y0 + size
    patch = np.full((size, size), 255, dtype=np.uint8)
    sx0 = max(0, x0); sy0 = max(0, y0)
    sx1 = min(w, x1); sy1 = min(h, y1)
    dx0 = sx0 - x0; dy0 = sy0 - y0
    if sx1 > sx0 and sy1 > sy0:
        patch[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = gray[sy0:sy1, sx0:sx1]
    # annulus mask
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2
    rr = np.hypot(xx - cx, yy - cy)
    ann = (rr >= (size * 0.18)) & (rr <= (size * 0.48))
    ring = np.where(ann, patch, 255).astype(np.float32)
    ring -= ring.mean()
    s = ring.std()
    if s > 1e-6:
        ring /= s
    return ring


def appearance_score(template, patch):
    if template is None or patch is None:
        return 0.0
    if template.shape != patch.shape:
        return 0.0
    corr = float((template * patch).sum()) / template.size
    corr = np.clip(corr, -1.0, 1.0)
    return (corr + 1.0) / 2.0


def make_colors(n):
    cmap = plt.get_cmap("tab20")
    cols = []
    for i in range(n):
        r, g, b, _ = cmap(i % 20)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


def main():
    files, frames = load_frames(FRAMES_GLOB)
    print("Loaded", len(frames), "frames")
    H0, W0 = frames[0].shape[:2]
    # unify sizes
    for i in range(len(frames)):
        if frames[i].shape[:2] != (H0, W0):
            frames[i] = cv2.resize(frames[i], (W0, H0))

    first_gray = ensure_gray(frames[0])
    mask = make_circle_mask((H0, W0), CIRCLE_CENTER, CIRCLE_RADIUS)

    det0 = detect_donuts(first_gray, mask)
    if det0.shape[0] == 0:
        raise RuntimeError("No donuts detected in first frame; adjust thresholds.")
    # pick nearest EXPECTED_TRACKS to center
    dists = np.hypot(det0[:, 0] - CIRCLE_CENTER[0], det0[:, 1] - CIRCLE_CENTER[1])
    idx = np.argsort(dists)[:EXPECTED_TRACKS]
    seeds = det0[idx].astype(float)
    if seeds.shape[0] < EXPECTED_TRACKS:
        pad = np.repeat(seeds[-1][np.newaxis, :], EXPECTED_TRACKS - seeds.shape[0], axis=0)
        seeds = np.vstack([seeds, pad])

    N = EXPECTED_TRACKS
    colors = make_colors(N)

    # create simple Kalman for motion
    kalmans = []
    for i in range(N):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[seeds[i, 0]], [seeds[i, 1]], [0.0], [0.0]], dtype=np.float32)
        kalmans.append(kf)

    templates = [extract_ring_template(first_gray, seeds[i, 0], seeds[i, 1], TEMPLATE_SIZE) for i in range(N)]
    missed = np.zeros(N, dtype=int)
    tracks = defaultdict(list)
    for pid in range(N):
        tracks[pid].append((0, float(seeds[pid, 0]), float(seeds[pid, 1])))

    prev_assigned_detection = np.full(N, -1, dtype=int)

    # preview
    vis0 = cv2.cvtColor(first_gray, cv2.COLOR_GRAY2BGR)
    for pid in range(N):
        x, y = int(round(seeds[pid, 0])), int(round(seeds[pid, 1]))
        cv2.circle(vis0, (x, y), 5, colors[pid], 2)
        cv2.putText(vis0, str(pid), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[pid], 1)
    cv2.circle(vis0, (int(CIRCLE_CENTER[0]), int(CIRCLE_CENTER[1])), int(CIRCLE_RADIUS), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "frame1_preview.png"), vis0)
    print("Saved frame1 preview")

    for fidx in range(1, len(frames)):
        gray = ensure_gray(frames[fidx])

        # predict
        preds = np.zeros((N, 2), dtype=float)
        for pid in range(N):
            p = kalmans[pid].predict()
            preds[pid, 0] = float(p[0, 0]); preds[pid, 1] = float(p[1, 0])

        dets = detect_donuts(gray, mask)
        M = dets.shape[0]
        print(f"Frame {fidx}: {M} detections")
        if M == 0:
            missed += 1
            continue
        dets_arr = np.array(dets, dtype=float)
        patches = [extract_ring_template(gray, dets_arr[j, 0], dets_arr[j, 1], TEMPLATE_SIZE) for j in range(M)]

        # detect close pairs to flag collisions
        det_pair_min = np.inf
        if M > 1:
            dmat = np.hypot(dets_arr[:, None, 0] - dets_arr[None, :, 0], dets_arr[:, None, 1] - dets_arr[None, :, 1])
            det_pair_min = np.min(dmat + np.eye(M) * 1e6)

        # cost matrix
        cost = np.full((N, M), 1e6, dtype=float)
        for i in range(N):
            for j in range(M):
                dist = np.hypot(preds[i, 0] - dets_arr[j, 0], preds[i, 1] - dets_arr[j, 1])
                if dist > MAX_DISP:
                    continue
                motion_score = max(0.0, 1.0 - dist / max(1.0, MAX_DISP))
                a_score = appearance_score(templates[i], patches[j])
                base_score = MOTION_WEIGHT * motion_score + APPEARANCE_WEIGHT * a_score
                switch_cost = 0.0
                if prev_assigned_detection[i] != -1 and prev_assigned_detection[i] != j:
                    switch_cost = SWITCH_PENALTY
                c = 1.0 - base_score + switch_cost
                cost[i, j] = c

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = np.full(N, -1, dtype=int)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0.95 and cost[r, c] < 1e5:
                dist = np.hypot(preds[r, 0] - dets_arr[c, 0], preds[r, 1] - dets_arr[c, 1])
                if dist <= MAX_DISP:
                    assigned[r] = c

        # resolve duplicates (rare)
        det_map = {}
        for pid in range(N):
            c = assigned[pid]
            if c >= 0:
                det_map.setdefault(c, []).append((pid, cost[pid, c]))
        for cidx, lst in det_map.items():
            if len(lst) > 1:
                lst.sort(key=lambda x: x[1])
                winner = lst[0][0]
                for pid, _ in lst[1:]:
                    assigned[pid] = -1

        matched = sum(1 for a in assigned if a >= 0)
        if fidx % 10 == 0 or matched < N:
            print(f"Frame {fidx}: matched {matched}/{N}")

        collision_present = (M > 1 and det_pair_min < COLLISION_DIST)

        # update tracks
        for pid in range(N):
            j = assigned[pid]
            if j >= 0:
                mx, my = float(dets_arr[j, 0]), float(dets_arr[j, 1])
                kalmans[pid].correct(np.array([[mx], [my]], dtype=np.float32))
                tracks[pid].append((fidx, mx, my))
                prev_assigned_detection[pid] = j
                missed[pid] = 0
                if not collision_present:
                    patch = patches[j]
                    if templates[pid] is None:
                        templates[pid] = patch
                    else:
                        templates[pid] = (1 - TEMPLATE_UPDATE_ALPHA) * templates[pid] + TEMPLATE_UPDATE_ALPHA * patch
            else:
                prev_assigned_detection[pid] = -1
                missed[pid] += 1

        # debug visualization
        if SAVE_DEBUG_FRAMES:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for j in range(M):
                xj, yj = int(round(dets_arr[j, 0])), int(round(dets_arr[j, 1]))
                cv2.circle(vis, (xj, yj), 6, (200, 200, 200), 1)
            for pid in range(N):
                px, py = int(round(preds[pid, 0])), int(round(preds[pid, 1]))
                cv2.circle(vis, (px, py), 3, (80, 80, 80), 1)
            for pid in range(N):
                j = assigned[pid]
                if j >= 0:
                    x, y = int(round(dets_arr[j, 0])), int(round(dets_arr[j, 1]))
                    cv2.circle(vis, (x, y), 6, colors[pid], -1)
                    cv2.putText(vis, str(pid), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.circle(vis, (int(CIRCLE_CENTER[0]), int(CIRCLE_CENTER[1])), int(CIRCLE_RADIUS), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "frames_debug", f"frame_{fidx:04d}.png"), vis)

    # build dataframe
    rows = []
    for pid in range(N):
        for fr, x, y in tracks[pid]:
            rows.append({"particle": int(pid), "frame": int(fr), "x": float(x), "y": float(y)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "tracked_particles.csv"), index=False)
    print("Saved tracked_particles.csv")

    # final plot
    plt.figure(figsize=(9, 9))
    plt.imshow(first_gray, cmap="gray")
    for pid in range(N):
        sub = df[df.particle == pid].sort_values("frame")
        if sub.shape[0] == 0:
            continue
        col = np.array(make_colors(N)[pid]) / 255.0
        plt.scatter(sub.x.values, sub.y.values, s=20, color=col, alpha=0.95)
    circ = plt.Circle(CIRCLE_CENTER, CIRCLE_RADIUS, fill=False, color="lime", lw=2)
    plt.gca().add_patch(circ)
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, "trajectory_map.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved trajectory_map.png")


if __name__ == "__main__":
    main()