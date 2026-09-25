import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.spatial import distance
import json

# ==============================
# 🔧 USER CONFIGURATION
# ==============================
input_folder = '/Users/tanishq/Desktop/untitled folder 8'   # <-- Your image folder
output_folder = '/Users/tanishq/Desktop/trajectory_output'   # <-- Output folder
os.makedirs(output_folder, exist_ok=True)

# Max distance (in pixels) a particle can move between frames
MAX_FRAME_DISTANCE = 30

# ==============================
# 🎯 FUNCTION: Extract blue centroids
# ==============================
def extract_blue_centroids(image):
    """
    Detects blue particles in an image and returns a list of their (x, y) centroids.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define blue range (tune if needed)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up the mask (remove small noise)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours and centroids
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
# 🚀 MAIN TRACKING LOGIC
# ==============================
print("Starting particle tracking...")

frames_paths = sorted(glob.glob(os.path.join(input_folder, "*.png")))
if not frames_paths:
    raise ValueError(f"No PNG files found in {input_folder}")

tracked_particles = {}       # {track_id: [(frame_index, (x, y)), ...]}
next_particle_id = 0
last_frame_particles = {}    # {track_id: (x, y)}

for frame_index, frame_path in enumerate(frames_paths):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"⚠️ Skipping unreadable frame: {frame_path}")
        continue

    current_centroids = extract_blue_centroids(frame)

    # First frame: initialize tracks
    if frame_index == 0:
        for (x, y) in current_centroids:
            tracked_particles[next_particle_id] = [(frame_index, (x, y))]
            last_frame_particles[next_particle_id] = (x, y)
            next_particle_id += 1
    else:
        current_frame_particles = {}
        last_ids = list(last_frame_particles.keys())
        last_positions = list(last_frame_particles.values())

        # Match existing tracks to new centroids
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

            # Add new particles (unmatched)
            for i, (x, y) in enumerate(current_centroids):
                if i not in used_current_centroids:
                    tracked_particles[next_particle_id] = [(frame_index, (x, y))]
                    current_frame_particles[next_particle_id] = (x, y)
                    next_particle_id += 1
        elif current_centroids:
            # No previous tracks, all are new
            for (x, y) in current_centroids:
                tracked_particles[next_particle_id] = [(frame_index, (x, y))]
                current_frame_particles[next_particle_id] = (x, y)
                next_particle_id += 1

        last_frame_particles = current_frame_particles

    if (frame_index + 1) % 100 == 0:
        print(f"Processed frame {frame_index + 1}/{len(frames_paths)}")

print(f"✅ Tracking complete — found {len(tracked_particles)} unique particle trajectories.")

# ==============================
# 🎨 FIXED TRAJECTORY PLOT (consistent colors)
# ==============================
print("Generating trajectory plot...")

plt.figure(figsize=(12, 10))

# Assign each track_id a unique, consistent color
track_ids = sorted(tracked_particles.keys())
cmap = plt.cm.get_cmap('tab20', len(track_ids))
color_map = {track_id: cmap(i % 20) for i, track_id in enumerate(track_ids)}

for track_id, path in tracked_particles.items():
    if len(path) < 2:
        continue

    frames, coords = zip(*path)
    xs, ys = zip(*coords)
    color = color_map[track_id]

    # Plot trajectory line
    plt.plot(xs, ys, color=color, alpha=0.8, linewidth=2)

    # Mark start and end points
    plt.scatter(xs[0], ys[0], color=color, s=100, marker='o', edgecolors='black')
    plt.scatter(xs[-1], ys[-1], color=color, s=100, marker='x')

plt.title("Particle Trajectories")
plt.xlabel("X position (px)")
plt.ylabel("Y position (px)")
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

out_plot_path = os.path.join(output_folder, "particle_trajectories_plot_fixed.png")
plt.savefig(out_plot_path, dpi=300)
plt.close()
print(f"✅ Fixed trajectory map saved at: {out_plot_path}")
