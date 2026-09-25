import os
import glob
import cv2
import numpy as np

# -------------------------- KEY PARAMETERS TO TUNE --------------------------
# Detection threshold (0–255). Lower = detects lighter gray particles.
DETECTION_THRESHOLD = 60
# Minimum size (in pixels) for valid particles
MIN_PARTICLE_AREA = 10
# ----------------------------------------------------------------------------

# ---------------- CONFIG ----------------
# Folder with your 1173 original filtered frames
INPUT_DIR = '/Users/tanishq/Desktop/filtered_frames'
# Folder where the detected (boxed) frames will be saved
OUTPUT_DIR = '/Users/tanishq/Desktop/Detected_Frames_Output'

# Visualization color for bounding boxes
BOX_COLOR = (255, 0, 0)  # Blue
BOX_THICKNESS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- Utilities ----------------
def list_images_sorted(folder):
    """Finds all images in a folder and sorts them numerically."""
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = [f for e in exts for f in glob.glob(os.path.join(folder, e))]
    return sorted(files)


# ---------------- Particle Detection Function ----------------
def detect_and_draw_particles(frame):
    """
    Detects dark particles within the inner white circular region.
    Avoids detecting the outer black border.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Isolate the main white circular region ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, circle_thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    # Clean up the circular region to remove specks or cracks
    circle_thresh = cv2.morphologyEx(circle_thresh, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Find largest contour (main white circular region)
    contours_circle, _ = cv2.findContours(circle_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_circle:
        return frame  # Skip if circle not found

    main_circle_contour = max(contours_circle, key=cv2.contourArea)

    # Create mask for interior of the circle (ignore the black edge)
    circle_mask = np.zeros_like(gray)
    cv2.drawContours(circle_mask, [main_circle_contour], -1, 255, thickness=cv2.FILLED)

    # Apply mask to grayscale image (keeps only inner region)
    gray_masked = cv2.bitwise_and(gray, gray, mask=circle_mask)

    # --- Detect dark particles within the white region ---
    _, thresh = cv2.threshold(gray_masked, DETECTION_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Apply the circular mask again just to be safe
    thresh = cv2.bitwise_and(thresh, thresh, mask=circle_mask)

    # Clean up small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours (particles)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame  # No particles detected

    # Filter contours by area and keep top 20 largest
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_PARTICLE_AREA]
    contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:20]

    # Draw bounding boxes around detected particles
    for cnt in contours_sorted:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

    return frame


# ---------------- Main Application Loop ----------------
def main():
    """Loops through all frames, detects particles, and saves results."""
    frames = list_images_sorted(INPUT_DIR)
    if not frames:
        print(f"Error: No images found in the input directory: {INPUT_DIR}")
        return

    for frame_idx, fp in enumerate(frames):
        frame = cv2.imread(fp)
        if frame is None:
            print(f"Warning: Could not read image file {fp}")
            continue

        # Detect and draw blue boxes
        output_frame = detect_and_draw_particles(frame)

        # Save output with original filename (no prefix)
        base_filename = os.path.basename(fp)
        out_path = os.path.join(OUTPUT_DIR, base_filename)
        cv2.imwrite(out_path, output_frame)

        print(f"Processed Frame {frame_idx + 1}/{len(frames)} -> Saved as {out_path}")

    print(f"\n✅ Success! All frames processed.")
    print(f"Frames with detected blue boxes saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
