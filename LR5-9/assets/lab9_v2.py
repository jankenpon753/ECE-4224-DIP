import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# Step 1: Read the image
img = cv2.imread("Letters.jpeg")
if img is None:
    raise FileNotFoundError("Could not read 'Letters.jpeg'. Please check the path.")

# Convert BGR to RGB for correct matplotlib display later
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 2: Pre-processing
if len(img.shape) == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img

# Binarize the image (using Otsu's thresholding for robust automatic calculation)
_, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Step 3: Select the pattern (Interactive)
print("Please select the ROI in the popped up window and press ENTER or SPACE.")
roi = cv2.selectROI(
    "Select Pattern (Press ENTER when done)", bw, showCrosshair=True, fromCenter=False
)
cv2.destroyWindow("Select Pattern (Press ENTER when done)")

# Extract coordinates and crop the pattern
x, y, w, h = roi
if w == 0 or h == 0:
    raise ValueError("No ROI selected. Exiting.")
pattern = bw[y : y + h, x : x + w]

# Step 4: Perform Template Matching
corr_map = cv2.matchTemplate(bw, pattern, cv2.TM_CCOEFF_NORMED)

# Step 5: Find peaks and apply Non-Maximum Suppression
threshold = 0.8

coordinates = peak_local_max(
    corr_map, min_distance=min(w, h) // 2, threshold_abs=threshold
)

# Step 6: Display Results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.canvas.manager.set_window_title("Pattern Detection Results")

# Subplot 1: The Template
axes[0].imshow(pattern, cmap="gray")
axes[0].set_title(f"Pattern to find (Size: {h}x{w})")
axes[0].axis("off")

# Subplot 2: Original Image with Detections
axes[1].imshow(img_rgb)
axes[1].set_title(f"Detected Locations: {len(coordinates)}")
axes[1].axis("off")

# Draw rectangles and center marks around every match found
for pt in coordinates:
    y_peak, x_peak = pt

    # Draw Rectangle
    rect = plt.Rectangle(
        (x_peak, y_peak), w, h, edgecolor="red", facecolor="none", linewidth=2
    )
    axes[1].add_patch(rect)

    # Draw Center Cross (+)
    center_x = x_peak + w / 2
    center_y = y_peak + h / 2
    axes[1].plot(center_x, center_y, "y+", markersize=10, markeredgewidth=2)

plt.tight_layout()
plt.show()

print(f"Finished! Found {len(coordinates)} instances of the pattern.")
