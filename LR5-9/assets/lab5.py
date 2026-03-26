import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Create a 5x5 grayscale matrix image (values 0-255)
img_exp5 = np.array(
    [
        [10, 50, 100, 150, 200],
        [30, 80, 120, 180, 220],
        [60, 110, 140, 210, 250],
        [90, 130, 170, 230, 240],
        [120, 160, 190, 245, 255],
    ],
    dtype=np.uint8,
)

T = 127  # Threshold value

# 2. Manual thresholding using nested loops
bw_img = np.zeros_like(img_exp5)
rows, cols = img_exp5.shape

for i in range(rows):
    for j in range(cols):
        if img_exp5[i, j] >= T:
            bw_img[i, j] = 1  # Or 255 for display purposes
        else:
            bw_img[i, j] = 0

print("--- Experiment 5: Thresholding ---")
print("Original Matrix:\n", img_exp5)
print(f"B&W Matrix (T={T}):\n", bw_img)
# Save the image
output_dir = "./images/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "lab5.png")
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_exp5, cmap="gray")
plt.title("Original Image")
plt.subplot(1, 2, 2)
plt.imshow(bw_img, cmap="gray")
plt.title(f"Thresholded (T={T})")
plt.savefig(output_path)
print(f"Image saved to {output_path}")
