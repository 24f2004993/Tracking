import cv2
import numpy as np
import glob
import os
import openpiv.tools
import openpiv.process
import openpiv.scaling

# Paths
input_folder = '/Users/tanishq/Desktop/untitled folder 3'
output_folder = '/Users/tanishq/Desktop/piv_collision_output'
os.makedirs(output_folder, exist_ok=True)

# Helper: Safe grayscale conversion
def to_gray(img):
    if len(img.shape) == 3 and img.shape[2] == 3:  # BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:  # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:  # Already grayscale
        return img

# Get all image files
frame_paths = sorted(glob.glob(os.path.join(input_folder, '*.*')))
print(f"Found {len(frame_paths)} frames")

# Process frame pairs
for i in range(len(frame_paths) - 1):
    frame_a = cv2.imread(frame_paths[i], cv2.IMREAD_UNCHANGED)
    frame_b = cv2.imread(frame_paths[i + 1], cv2.IMREAD_UNCHANGED)

    if frame_a is None or frame_b is None:
        print(f"Skipping frames {frame_paths[i]} or {frame_paths[i+1]} - not readable")
        continue

    # Convert to grayscale safely
    gray_a = to_gray(frame_a)
    gray_b = to_gray(frame_b)

    # Perform PIV
    u, v, sig2noise = openpiv.process.extended_search_area_piv(
        gray_a.astype(np.int32),
        gray_b.astype(np.int32),
        window_size=32,
        overlap=16,
        dt=1,
        search_area_size=32,
        sig2noise_method='peak2peak'
    )

    x, y = openpiv.process.get_coordinates(
        image_size=gray_a.shape,
        window_size=32,
        overlap=16
    )

    u, v, mask = openpiv.validation.sig2noise_val(u, v, sig2noise, threshold=1.3)
    u, v = openpiv.filters.replace_outliers(u, v, method='localmean', max_iter=3, kernel_size=2)

    # Scale to pixel units
    u, v, x, y = openpiv.scaling.uniform(u, v, x, y, scaling_factor=1)

    # Draw arrows on the original frame
    vis_frame = frame_a.copy()
    step = 1  # draw every vector
    for yy in range(0, u.shape[0], step):
        for xx in range(0, u.shape[1], step):
            start_point = (int(x[yy, xx]), int(y[yy, xx]))
            end_point = (int(x[yy, xx] + u[yy, xx]), int(y[yy, xx] + v[yy, xx]))
            cv2.arrowedLine(vis_frame, start_point, end_point, (0, 0, 255), 1, tipLength=0.3)

    # Save output
    out_path = os.path.join(output_folder, f"piv_{i:04d}.png")
    cv2.imwrite(out_path, vis_frame)
    print(f"Saved: {out_path}")

print("PIV processing completed.")