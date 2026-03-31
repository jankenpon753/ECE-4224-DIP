import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

# 1. Define Image and Structuring Element (SE)
img_exp7 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

SE = np.ones((3, 3), dtype=np.uint8)


# 2. Manual Erosion (min of window)
def manual_erode(img, se):
    pad = se.shape[0] // 2
    # Padding with 0 as requested in the prompt
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i : i + se.shape[0], j : j + se.shape[1]]
            out[i, j] = np.min(window)
    return out


# 3. Manual Dilation (max of window)
def manual_dilate(img, se):
    pad = se.shape[0] // 2
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i : i + se.shape[0], j : j + se.shape[1]]
            out[i, j] = np.max(window)
    return out


# Using scipy imdilate equivalent
# dilated_img_scipy = binary_dilation(img_exp7, structure=SE).astype(np.uint8)
# eroded_img_scipy = binary_erosion(img_exp7, structure=SE).astype(np.uint8)


eroded_img = manual_erode(img_exp7, SE)
dilated_img = manual_dilate(img_exp7, SE)

print("\n--- Experiment 7: Morphological Operations ---")
print("Original 7x7:\n", img_exp7)
print("Dilated (Expands 1s):\n", dilated_img)
print("Eroded (Shrinks 1s):\n", eroded_img)
# print("Eroded (Shrinks 1s):\n", eroded_img_scipy)
# print("Dilated (Expands 1s):\n", dilated_img_scipy)

# Save the image
output_dir = "./images/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "lab7.png")

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_exp7, cmap="gray")
plt.title("Original Image")
# Add gridlines
ax1 = plt.gca()
ax1.set_xticks(np.arange(-0.5, img_exp7.shape[1], 1), minor=True)
ax1.set_yticks(np.arange(-0.5, img_exp7.shape[0], 1), minor=True)
ax1.grid(which="minor", color="red", linestyle="-", linewidth=0.5)
plt.subplot(1, 3, 2)
plt.imshow(dilated_img, cmap="gray", vmin=0, vmax=1)
plt.title("Dilated Image")
# Add gridlines
ax2 = plt.gca()
ax2.set_xticks(np.arange(-0.5, dilated_img.shape[1], 1), minor=True)
ax2.set_yticks(np.arange(-0.5, dilated_img.shape[0], 1), minor=True)
ax2.grid(which="minor", color="red", linestyle="-", linewidth=0.5)
plt.subplot(1, 3, 3)
plt.imshow(np.round(eroded_img).astype(np.uint8), cmap="gray")
plt.title("Eroded Image")
# Add gridlines
ax3 = plt.gca()
ax3.set_xticks(np.arange(-0.5, eroded_img.shape[1], 1), minor=True)
ax3.set_yticks(np.arange(-0.5, eroded_img.shape[0], 1), minor=True)
ax3.grid(which="minor", color="red", linestyle="-", linewidth=0.5)
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Image saved to {output_path}")
plt.show()
