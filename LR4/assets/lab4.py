import numpy as np
import matplotlib.pyplot as plt
import math
import os

os.makedirs("./images/output", exist_ok=True)
# 1. Define a 4x4 Matrix "Image" Using a checkerboard-like pattern with distinct values for clarity
# 10 = Dark, 200 = Bright
img_4x4 = np.array(
    [[10, 200, 10, 200], [200, 50, 200, 50], [10, 200, 10, 200], [200, 50, 200, 50]],
    dtype=np.uint8,
)

m, n = img_4x4.shape
s = 3  # Scaling Factor

# Method 1: Nearest Neighbor (Manual Loop)
new_m = m * s
new_n = n * s
zoomed_nn = np.zeros((new_m, new_n), dtype=np.uint8)

# Loop through original 4x4 pixels
for i in range(m):
    for j in range(n):
        val = img_4x4[i, j]

        # Fill the 3x3 block in the new image
        # Row start: i*3, Col start: j*3
        for r in range(s):
            for c in range(s):
                zoomed_nn[i * s + r, j * s + c] = val

# Method 2: Bilinear Interpolation (Manual Loop)
zoomed_bl = np.zeros((new_m, new_n), dtype=np.uint8)


# Helper function to get pixel safely (clamping to edges)
def get_pixel(img, x, y):
    h, w = img.shape
    x = min(max(x, 0), h - 1)
    y = min(max(y, 0), w - 1)
    return img[x, y]


# Loop through NEW 8x8 pixels
for i in range(new_m):
    for j in range(new_n):

        # Map back to original coordinate space (i / s) scales 0..7 back to 0..3.5 range
        orig_x = i / s
        orig_y = j / s

        # Find 4 Nearest Neighbors
        x1 = int(math.floor(orig_x))
        y1 = int(math.floor(orig_y))
        x2 = x1 + 1
        y2 = y1 + 1

        # Calculate distances (0 to 1)
        alpha = orig_x - x1
        beta = orig_y - y1

        # Get values of neighbors
        Q11 = get_pixel(img_4x4, x1, y1)  # Top-Left
        Q21 = get_pixel(img_4x4, x2, y1)  # Bottom-Left
        Q12 = get_pixel(img_4x4, x1, y2)  # Top-Right
        Q22 = get_pixel(img_4x4, x2, y2)  # Bottom-Right

        # Bilinear Formula
        val = (
            (1 - alpha) * (1 - beta) * Q11
            + alpha * (1 - beta) * Q21
            + (1 - alpha) * beta * Q12
            + alpha * beta * Q22
        )

        zoomed_bl[i, j] = int(val)


# Visualization
def plot_matrix(ax, matrix, title):
    ax.imshow(matrix, cmap="viridis", vmin=0, vmax=255)
    ax.set_title(title)
    # Draw grid lines to show pixels clearly
    h, w = matrix.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)
    # Annotate values
    for i in range(h):
        for j in range(w):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] < 150 else "black",
                fontsize=8,
            )


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_matrix(axes[0], img_4x4, "Original (4x4)")
plot_matrix(axes[1], zoomed_nn, "Nearest Neighbor (8x8)")
plot_matrix(axes[2], zoomed_bl, "Bilinear (8x8)")

plt.tight_layout()
plt.savefig("./images/output/lab4_color.png")
# plt.show()

# Print Matrices to Console for Report Verification
print("--- Original 4x4 ---")
print(img_4x4)
print("\n--- Nearest Neighbor 8x8 (Note the 2x2 blocks) ---")
print(zoomed_nn)
print("\n--- Bilinear 8x8 (Note the smoothing) ---")
print(zoomed_bl)
