import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Define the structuring element
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


# 1. Define Opening and Closing functions using Exp 7 definitions
def manual_open(img, se):
    # Opening = Erode -> Dilate
    return manual_dilate(manual_erode(img, se), se)


def manual_close(img, se):
    # Closing = Dilate -> Erode
    return manual_erode(manual_dilate(img, se), se)


# We use an image with noise to show the effect
img_exp8 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)


opened_img = manual_open(img_exp8, SE)
closed_img = manual_close(img_exp8, SE)

print("\n--- Experiment 8: Opening & Closing ---")
print("Original Image (with sticking out noise):\n", img_exp8)
print("Opened Image (Removes noise):\n", opened_img)
print("Closed Image (Fills holes):\n", closed_img)
# Save the image
output_dir = "./images/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "lab8.png")

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_exp8, cmap="gray")
plt.title("Original Image")
# Add gridlines
ax1 = plt.gca()
ax1.set_xticks(np.arange(-0.5, img_exp8.shape[1], 1), minor=True)
ax1.set_yticks(np.arange(-0.5, img_exp8.shape[0], 1), minor=True)
ax1.grid(which="minor", color="red", linestyle="-", linewidth=0.5)
plt.subplot(1, 3, 2)
plt.imshow(opened_img, cmap="gray")
plt.title("Opened Image")
# Add gridlines
ax2 = plt.gca()
ax2.set_xticks(np.arange(-0.5, opened_img.shape[1], 1), minor=True)
ax2.set_yticks(np.arange(-0.5, opened_img.shape[0], 1), minor=True)
ax2.grid(which="minor", color="red", linestyle="-", linewidth=0.5)
plt.subplot(1, 3, 3)
plt.imshow(closed_img, cmap="gray")
plt.title("Closed Image")
# Add gridlines
ax3 = plt.gca()
ax3.set_xticks(np.arange(-0.5, closed_img.shape[1], 1), minor=True)
ax3.set_yticks(np.arange(-0.5, closed_img.shape[0], 1), minor=True)
ax3.grid(which="minor", color="red", linestyle="-", linewidth=0.5)
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Image saved to {output_path}")
plt.show()
