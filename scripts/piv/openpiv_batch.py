import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from openpiv import tools, scaling, validation, filters, pyprocess

# === PATHS ===
INPUT_DIR = Path("/Users/tanishq/Desktop/untitled folder 3")
OUTPUT_DIR = Path("/Users/tanishq/Desktop/piv_collision_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === PIV PARAMETERS ===
WINDOW_SIZE = 32
OVERLAP = 16
DT = 1.0
SIG2NOISE_THRESHOLD = 1.3

# === LOAD & SORT IMAGES ===
images = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']],
                key=lambda x: int(''.join(filter(str.isdigit, x.name))))

if len(images) < 2:
    raise ValueError("At least 2 frames are required for PIV.")

# === MAIN LOOP ===
for i in tqdm(range(len(images)-1), desc="Processing PIV"):
    frame_a = tools.imread(str(images[i]))
    frame_b = tools.imread(str(images[i+1]))

    # Convert to grayscale
    if frame_a.ndim == 3:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    else:
        gray_a, gray_b = frame_a, frame_b

    # Compute PIV
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        dt=DT,
        search_area_size=WINDOW_SIZE,
        sig2noise_method='peak2peak'
    )

    # Validation: get mask of valid vectors
    mask = validation.sig2noise_val(sig2noise, threshold=SIG2NOISE_THRESHOLD)

    # Replace outliers
    u, v = filters.replace_outliers(u, v, method='localmean', max_iter=3, kernel_size=2)

    # Apply mask to u, v
    u = u * mask
    v = v * mask

    # Coordinates
    x, y = tools.get_coordinates(image_size=gray_a.shape, window_size=WINDOW_SIZE, overlap=OVERLAP)

    # Scaling
    u, v = scaling.uniform(u, v, scaling_factor=1)

    # Overlay arrows (blue)
    overlay = cv2.cvtColor(gray_a, cv2.COLOR_GRAY2BGR)
    for xi, yi, ui, vi in zip(x.flatten(), y.flatten(), u.flatten(), v.flatten()):
        start = (int(xi), int(yi))
        end = (int(xi + ui*1.5), int(yi + vi*1.5))
        cv2.arrowedLine(overlay, start, end, (255, 0, 0), 2, tipLength=0.3)

    # Save output
    out_path = OUTPUT_DIR / f"piv_{i+1:04d}.png"
    cv2.imwrite(str(out_path), overlay)

print(f"\n✅ PIV processing done. Check results in: {OUTPUT_DIR}")