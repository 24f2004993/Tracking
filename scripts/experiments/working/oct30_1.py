import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.spatial import distance

# ==============================
# 🔧 USER CONFIGURATION
# ==============================
input_folder = '/Users/tanishq/Desktop/untitled folder 8'      # clean frames
annotated_folder = '/Users/tanishq/Desktop/annotated_frames'    # annotated frames (with red dots)
output_folder = '/Users/tanishq/Desktop/trajectory_output'
debug_output = os.path.join(output_folder, "debug_frames")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_output, exist_ok=True)

MAX_FRAME_DISTANCE = 30  # max px distance per frame to link same particle

# ==============================
# 🎯 FUNCTION: Extract blue centroids
# ==============================
def extract_blue_centroids(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    return centroids

# ==============================
# 🚀 TRACK BLUE PARTICLES
# ==============================
frames_paths = sorted(glob.glob(os.path.join(input_folder, "*.png")))
if not frames_paths:
    raise ValueError(f"No PNG files found in {input_folder}")

tracked_particles = {}
next_particle_id = 0
last_frame_particles = {}

print("Starting particle tracking...")

for frame_index, frame_path in enumerate(frames_paths):
    frame = cv2.imread(frame_path)
    if frame is None:
        continue
    current_centroids = extract_blue_centroids(frame)

    if frame_index == 0:
        for (x, y) in current_centroids:
            tracked_particles[next_particle_id] = [(frame_index, (x, y))]
            last_frame_particles[next_particle_id] = (x, y)
            next_particle_id += 1
    else:
        current_frame_particles = {}
        last_ids = list(last_frame_particles.keys())
        last_positions = list(last_frame_particles.values())

        if last_positions and current_centroids:
            dist_matrix = distance.cdist(last_positions, current_centroids, 'euclidean')
            matches = []
            for i in range(len(last_ids)):
                for j in range(len(current_centroids)):
                    matches.append((dist_matrix[i, j], i, j))
            matches.sort()
            used_last_ids = set()
            used_current_centroids = set()

            for dist, last_idx, current_idx in matches:
                if dist > MAX_FRAME_DISTANCE:
                    break
                if last_idx in used_last_ids or current_idx in used_current_centroids:
                    continue
                track_id = last_ids[last_idx]
                (x, y) = current_centroids[current_idx]
                tracked_particles[track_id].append((frame_index, (x, y)))
                current_frame_particles[track_id] = (x, y)
                used_last_ids.add(last_idx)
                used_current_centroids.add(current_idx)

            for i, (x, y) in enumerate(current_centroids):
                if i not in used_current_centroids:
                    tracked_particles[next_particle_id] = [(frame_index, (x, y))]
                    current_frame_particles[next_particle_id] = (x, y)
                    next_particle_id += 1
        elif current_centroids:
            for (x, y) in current_centroids:
                tracked_particles[next_particle_id] = [(frame_index, (x, y))]
                current_frame_particles[next_particle_id] = (x, y)
                next_particle_id += 1

        last_frame_particles = current_frame_particles

print(f"✅ Tracking complete — found {len(tracked_particles)} unique particles.")

# ==============================
# 🔴 COLLISION DETECTION FROM RED DOTS
# ==============================
collision_frames = {}
annotated_paths = sorted(glob.glob(os.path.join(annotated_folder, "*.png")))
print("Detecting collisions from red dots...")

# Define HSV range for red
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

for frame_index, anno_path in enumerate(annotated_paths):
    if frame_index >= len(frames_paths):
        break

    anno_img = cv2.imread(anno_path)
    if anno_img is None:
        continue

    hsv = cv2.cvtColor(anno_img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_dots = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 3 <= w <= 15 and 3 <= h <= 15:
            red_dots.append((x + w // 2, y + h // 2))

    collided_ids = []
    for pid, path in tracked_particles.items():
        for f_idx, (x, y) in path:
            if f_idx == frame_index:
                if any(np.hypot(x - rx, y - ry) < 12 for (rx, ry) in red_dots):
                    collided_ids.append(pid)
                    break

    if collided_ids:
        collision_frames[frame_index] = collided_ids
        print(f"💥 Frame {frame_index}: {len(collided_ids)} collisions detected")

print(f"✅ Collision detection done. Found collisions in {len(collision_frames)} frames.")

# ==============================
# 🎨 COLORFUL DOT TRAJECTORY MAP
# ==============================
print("Generating colorful trajectory map with collision dots...")

plt.figure(figsize=(12, 10))
track_ids = sorted(tracked_particles.keys())
colors = plt.cm.hsv(np.linspace(0, 1, len(track_ids)))
color_map = {tid: colors[i] for i, tid in enumerate(track_ids)}

for tid, path in tracked_particles.items():
    for (frame_idx, (x, y)) in path:
        if frame_idx in collision_frames and tid in collision_frames[frame_idx]:
            plt.scatter(x, y, color='red', s=40, alpha=1)
        else:
            plt.scatter(x, y, color=color_map[tid], s=40, alpha=0.8)

plt.title("Particle Trajectories with Collisions (Red Dots)")
plt.xlabel("X position (px)")
plt.ylabel("Y position (px)")
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

out_plot_path = os.path.join(output_folder, "particle_trajectories_with_collisions.png")
plt.savefig(out_plot_path, dpi=300)
plt.close()

print(f"✅ Trajectory map saved at: {out_plot_path}")
print(f"🧩 Debug frames saved at: {debug_output}")
