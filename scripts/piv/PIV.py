import os
from pathlib import Path
import numpy as np
import cv2
from skimage.io import imread
from skimage.feature import blob_log
from tqdm import tqdm

# === PATHS ===
IMG_DIR = Path("/Users/tanishq/Desktop/10_frames")
OUT_DIR = Path("/Users/tanishq/Desktop/10_frames_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === PARAMETERS ===
BLOB_MIN_SIGMA = 3
BLOB_MAX_SIGMA = 6
BLOB_THRESH = 0.05
MAX_JUMP = 20
ARROW_SCALE = 1.0
DUPLICATE_DIST = 5

# === LOAD & SORT IMAGE FRAMES ===
images = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']],
                key=lambda x: int(''.join(filter(str.isdigit, x.name))))
if len(images) < 2:
    raise ValueError("At least 2 frames are required")

# === DETECTION FUNCTION ===
def detect_particles(img):
    blobs = blob_log(img,
                     min_sigma=BLOB_MIN_SIGMA,
                     max_sigma=BLOB_MAX_SIGMA,
                     threshold=BLOB_THRESH)
    return blobs[:, :2][:, ::-1]  # (x, y)

# === PROCESS EACH FRAME AND DRAW ARROWS ON CURRENT FRAME ===
for i in tqdm(range(len(images) - 1), desc="Tracking"):
    # Load images
    img1 = imread(images[i])
    img2 = imread(images[i + 1])

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2

    # Detect particles
    pts1 = detect_particles(gray1)
    pts2 = detect_particles(gray2)

    matches = []
    used_pts2 = []

    for p1 in pts1:
        dists = np.linalg.norm(pts2 - p1, axis=1)
        if len(dists) == 0:
            continue
        min_idx = np.argmin(dists)
        p2 = pts2[min_idx]

        if dists[min_idx] > MAX_JUMP:
            continue

        if any(np.linalg.norm(p2 - u) < DUPLICATE_DIST for u in used_pts2):
            continue

        matches.append((p1, p2))
        used_pts2.append(p2)

    # Draw arrows on the original image (img1)
    overlay = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)

    for (x1, y1), (x2, y2) in matches:
        dx, dy = (x2 - x1) * ARROW_SCALE, (y2 - y1) * ARROW_SCALE
        start = (int(x1), int(y1))
        end = (int(x1 + dx), int(y1 + dy))
        cv2.arrowedLine(overlay, start, end, (0, 0, 255), 2, tipLength=0.3)

    # Save the output with arrow drawn on original frame
    out_path = OUT_DIR / f"motion_{i+1:04d}.png"
    cv2.imwrite(str(out_path), overlay)

print(f"\n✅ Done! Arrows drawn on original frames. Check: {OUT_DIR}")