import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from openpiv import tools, validation, filters, scaling, piv

# === PATHS ===
INPUT_DIR = Path("data/collision_frames")
OUTPUT_DIR = Path("outputs/piv_collision")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === PIV PARAMETERS ===
WINDOW_SIZE = 32          # interrogation window size
OVERLAP = 16              # overlap between windows
SIG2NOISE_THRESHOLD = 1.3 # validation threshold

# === LOAD & SORT IMAGES ===
images = sorted(
    [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']],
    key=lambda x: int(''.join(filter(str.isdigit, x.name)))
)

if len(images) < 2:
    raise ValueError("At least 2 frames are required for PIV.")

# === MAIN LOOP ===
for i in tqdm(range(len(images)-1), desc="Processing PIV"):
    frame_a = tools.imread(str(images[i]))
    frame_b = tools.imread(str(images[i+1]))

    # Convert to grayscale if needed
    if frame_a.ndim == 3:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    else:
        gray_a, gray_b = frame_a, frame_b

    # Compute PIV (note: simple_piv expects positional args: frame_a, frame_b, window_size, overlap)
    u, v, sig2noise = piv.simple_piv(
        gray_a.astype(np.int32),
        gray_b.astype(np.int32),
        WINDOW_SIZE,
        OVERLAP,
        method='peak2peak'
    )

    # Validation
    mask = validation.sig2noise_val(sig2noise, threshold=SIG2NOISE_THRESHOLD)
    u, v = filters.replace_outliers(u, v, mask, method='localmean', max_iter=3, kernel_size=2)

    # Coordinates
    x, y = tools.get_coordinates(image_size=gray_a.shape, window_size=WINDOW_SIZE, overlap=OVERLAP)

    # Scaling
    u, v = scaling.uniform(u, v, scaling_factor=1)

    # Overlay arrows
    overlay = cv2.cvtColor(gray_a, cv2.COLOR_GRAY2BGR)
    for xi, yi, ui, vi, m in zip(x.flatten(), y.flatten(), u.flatten(), v.flatten(), mask.flatten()):
        if m:  # only draw validated vectors
            start = (int(xi), int(yi))
            end = (int(xi + ui*1.5), int(yi + vi*1.5))
            cv2.arrowedLine(overlay, start, end, (255, 0, 0), 2, tipLength=0.3)

    # Save output
    out_path = OUTPUT_DIR / f"piv_{i+1:04d}.png"
    cv2.imwrite(str(out_path), overlay)

print(f"\n✅ PIV processing done. Check results in: {OUTPUT_DIR}")
