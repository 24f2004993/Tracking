import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import pims
import cv2
import os

# ----------------- SETTINGS -----------------
FRAMES_PATH = "/Users/tanishq/Desktop/New Folder With Items/*.jpg"
OUTPUT_DIR  = "/Users/tanishq/Desktop/particle_tracking_output1"

DIAMETER = 31
MINMASS = 1800
SEARCH_RANGE = 25
MEMORY = 5
COLLISION_DIST = 45

CIRCLE_CENTER = (500, 500)
CIRCLE_RADIUS = 477
# ------------------------------------------------


def inside_circle(x, y, center, radius):
    cx, cy = center
    return (x - cx)**2 + (y - cy)**2 <= radius**2


def preprocess(gray):
    """Enhance contrast slightly for stable detection."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    alpha, beta = 1.25, 10
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)


def ensure_output_dir(path):
    """Ensure output directory exists and is writable."""
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        print(f"✅ Output directory ready: {path}")
    except Exception as e:
        print(f"❌ Cannot write to {path}: {e}")
        exit(1)


def main():
    ensure_output_dir(OUTPUT_DIR)

    frames = pims.open(FRAMES_PATH)
    print(f"\nLoaded {len(frames)} frames\n")

    # -------- DETECTION ON FIRST FRAME --------
    print("Testing detection on first frame...")
    first_frame = frames[0]
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY) if len(first_frame.shape) == 3 else first_frame
    processed_gray = preprocess(first_gray)

    test_df = tp.locate(processed_gray, diameter=DIAMETER, minmass=MINMASS, invert=True)
    test_df = test_df[test_df.apply(lambda r: inside_circle(r['x'], r['y'], CIRCLE_CENTER, CIRCLE_RADIUS), axis=1)]
    print(f"Particles detected inside circle: {len(test_df)}")

    plt.figure(figsize=(7,7))
    plt.imshow(first_gray, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    plt.scatter(test_df['x'], test_df['y'], edgecolors='red', facecolors='none', s=80, linewidths=1.5)
    circ = plt.Circle(CIRCLE_CENTER, CIRCLE_RADIUS, color='lime', fill=False, lw=2)
    plt.gca().add_patch(circ)
    plt.axis('off')
    plt.title("Verification: Frame 1 Detections")
    plt.tight_layout()

    test_img_path = os.path.join(OUTPUT_DIR, "frame1_detections.png")
    plt.savefig(test_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved detection preview: {test_img_path}")

    # -------- DETECT IN ALL FRAMES --------
    print("\nDetecting and linking across all frames...")
    all_detections = []
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        proc = preprocess(gray)
        df = tp.locate(proc, diameter=DIAMETER, minmass=MINMASS, invert=True)
        if df is not None and len(df) > 0:
            df['frame'] = i
            df = df[df.apply(lambda r: inside_circle(r['x'], r['y'], CIRCLE_CENTER, CIRCLE_RADIUS), axis=1)]
            all_detections.append(df)
        if i % 100 == 0:
            print(f"Processed {i}/{len(frames)} frames...")

    f = pd.concat(all_detections, ignore_index=True)
    print(f"\nTotal detections inside circle: {len(f)}")

    # -------- LINK PARTICLES ACROSS FRAMES --------
    print("Linking trajectories...")
    tracks = tp.link_df(f, search_range=SEARCH_RANGE, memory=MEMORY)
    print(f"Particles tracked: {tracks['particle'].nunique()}")

    # Filter short trajectories
    tracks = tp.filter_stubs(tracks, threshold=10)
    print(f"Particles after filtering: {tracks['particle'].nunique()}")

    # Save tracking data
    tracks_file = os.path.join(OUTPUT_DIR, "tracked_particles.csv")
    tracks.to_csv(tracks_file, index=False)
    print(f"✅ Saved tracking data: {tracks_file}")

    # -------- DOT-BASED TRAJECTORY MAP --------
    print("\nGenerating trajectory map (dots)...")

    plt.figure(figsize=(9,9))
    plt.imshow(first_gray, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    circ = plt.Circle(CIRCLE_CENTER, CIRCLE_RADIUS, color='lime', fill=False, lw=2)
    plt.gca().add_patch(circ)

    # Each particle gets its own color and dots instead of lines
    unique_particles = tracks['particle'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_particles)))

    for color, pid in zip(colors, unique_particles):
        sub = tracks[tracks['particle'] == pid]
        plt.scatter(sub['x'], sub['y'], s=10, color=color, alpha=0.9)  # dots only

    plt.axis('off')
    plt.title("Trajectory Map (Dots for Each Frame)")
    plt.tight_layout()

    traj_file = os.path.join(OUTPUT_DIR, "trajectory_map_dots.png")
    plt.savefig(traj_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved trajectory map (dots): {traj_file}")

    # -------- COLLISION DETECTION --------
    print("\nDetecting possible collisions...")
    tracks = tracks.reset_index(drop=True)  # 🟢 FIX: removes index ambiguity

    collisions = []
    for frame, group in tracks.groupby("frame"):
        pts = group[['particle','x','y']].values
        for i in range(len(pts)):
            id1, x1, y1 = pts[i]
            for j in range(i+1, len(pts)):
                id2, x2, y2 = pts[j]
                d = np.hypot(x1-x2, y1-y2)
                if d < COLLISION_DIST:
                    collisions.append([frame, int(id1), int(id2), float(d)])

    collisions = pd.DataFrame(collisions, columns=["frame","id1","id2","distance"])
    collisions_file = os.path.join(OUTPUT_DIR, "collisions.csv")
    collisions.to_csv(collisions_file, index=False)
    print(f"✅ Saved collisions: {collisions_file}")

    print("\n🎉 DONE — All results saved successfully in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()