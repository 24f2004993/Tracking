import os
import glob
import cv2
import numpy as np

# ---------------- CONFIG ----------------
INPUT_DIR = '/Users/tanishq/Desktop/filtered_frames'
OUTPUT_DIR = '/Users/tanishq/Desktop/Detected_Frames_Output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualization color
BOX_COLOR = (255, 0, 0)
BOX_THICKNESS = 2

# ---------------- Utilities ----------------
def list_images_sorted(folder):
    exts = ("*.png", "*.jpg", "*.jpeg")
    files = [f for e in exts for f in glob.glob(os.path.join(folder, e))]
    return sorted(files)

# ---------------- Detection ----------------
def detect_particles(frame, threshold, min_area, top_n):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, circle_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours_circle, _ = cv2.findContours(circle_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_circle:
        return frame
    main_circle_contour = max(contours_circle, key=cv2.contourArea)
    circle_mask = np.zeros_like(gray)
    cv2.drawContours(circle_mask, [main_circle_contour], -1, 255, cv2.FILLED)
    gray_masked = cv2.bitwise_and(gray, gray, mask=circle_mask)
    _, thresh = cv2.threshold(gray_masked, threshold, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_and(thresh, thresh, mask=circle_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    contours_sorted = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:top_n]
    output = frame.copy()
    for cnt in contours_sorted:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
    return output

# ---------------- Interactive Simulation ----------------
def interactive_tuning(sample_frame_path):
    frame = cv2.imread(sample_frame_path)
    cv2.namedWindow('Tuning')

    def nothing(x):
        pass

    cv2.createTrackbar('Threshold', 'Tuning', 60, 255, nothing)
    cv2.createTrackbar('Min Area', 'Tuning', 10, 500, nothing)
    cv2.createTrackbar('Top N', 'Tuning', 20, 100, nothing)

    print("\n🎛️ Adjust sliders live. Press 's' to save settings and process all frames.")
    print("Press 'q' to quit without saving.\n")

    while True:
        threshold = cv2.getTrackbarPos('Threshold', 'Tuning')
        min_area = cv2.getTrackbarPos('Min Area', 'Tuning')
        top_n = cv2.getTrackbarPos('Top N', 'Tuning')

        display = detect_particles(frame, threshold, min_area, top_n)
        cv2.imshow('Tuning', display)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('s'):
            cv2.destroyAllWindows()
            return threshold, min_area, top_n
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

# ---------------- Batch Processing ----------------
def process_all_frames(threshold, min_area, top_n):
    frames = list_images_sorted(INPUT_DIR)
    if not frames:
        print(f"Error: No images found in {INPUT_DIR}")
        return

    print(f"\n▶️ Processing {len(frames)} frames with Threshold={threshold}, MinArea={min_area}, TopN={top_n}\n")
    for i, fp in enumerate(frames):
        frame = cv2.imread(fp)
        if frame is None:
            continue
        output = detect_particles(frame, threshold, min_area, top_n)
        base = os.path.basename(fp)
        out_path = os.path.join(OUTPUT_DIR, f"detected_{base}")
        cv2.imwrite(out_path, output)
        print(f"Processed {i+1}/{len(frames)} -> {out_path}")
    print("\n✅ All frames processed and saved.")

# ---------------- Main ----------------
def main():
    frames = list_images_sorted(INPUT_DIR)
    if not frames:
        print(f"Error: No images found in {INPUT_DIR}")
        return
    sample_frame = frames[0]
    result = interactive_tuning(sample_frame)
    if result:
        threshold, min_area, top_n = result
        process_all_frames(threshold, min_area, top_n)
    else:
        print("\n❌ Exited without saving.")

if __name__ == '__main__':
    main()