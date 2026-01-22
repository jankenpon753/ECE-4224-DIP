import numpy as np
import matplotlib.pyplot as plt
import math
import os

os.makedirs("./images/output", exist_ok=True)
# 1. Define a 4x4 Matrix "Image"
# CHANGED: Used 0 and 255 instead of 0 and 1.
# Reason: int() truncation on 0-1 range destroys bilinear gradients (e.g., int(0.5) = 0).
# 0 = Black, 255 = White
img_4x4 = np.array(
    [[0, 255, 0, 255], [255, 0, 255, 0], [0, 255, 0, 255], [255, 0, 255, 0]],
    dtype=np.uint8,
)

m, n = img_4x4.shape
s = 4  # Scaling Factor

# ==========================================
# Method 1: Nearest Neighbor (Manual Loop)
# ==========================================
new_m = m * s
new_n = n * s
zoomed_nn = np.zeros((new_m, new_n), dtype=np.uint8)

# Loop through original 4x4 pixels
for i in range(m):
    for j in range(n):
        val = img_4x4[i, j]

        # Fill the 3x3 block in the new image
        for r in range(s):
            for c in range(s):
                zoomed_nn[i * s + r, j * s + c] = val

# ==========================================
# Method 2: Bilinear Interpolation (Manual Loop)
# ==========================================
zoomed_bl = np.zeros((new_m, new_n), dtype=np.uint8)


# Helper function to get pixel safely (clamping to edges)
def get_pixel(img, x, y):
    h, w = img.shape
    x = min(max(x, 0), h - 1)
    y = min(max(y, 0), w - 1)
    return img[x, y]


# Loop through NEW pixels
for i in range(new_m):
    for j in range(new_n):

        # Map back to original coordinate space
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


# ==========================================
# Visualization
# ==========================================
def plot_matrix(ax, matrix, title):
    # CHANGED: cmap='gray' for monochrome
    ax.imshow(matrix, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)

    # Grid lines
    h, w = matrix.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Annotate values
    for i in range(h):
        for j in range(w):
            val = matrix[i, j]
            # Dynamic text color for visibility
            text_color = "white" if val < 127 else "black"
            ax.text(
                j, i, str(val), ha="center", va="center", color=text_color, fontsize=6
            )


fig, axes = plt.subplots(1, 3, figsize=(15, 6))

plot_matrix(axes[0], img_4x4, "Original (4x4)")
plot_matrix(axes[1], zoomed_nn, f"Nearest Neighbor ({new_m}x{new_n})")
plot_matrix(axes[2], zoomed_bl, f"Bilinear ({new_m}x{new_n})")

plt.tight_layout()
plt.savefig("./images/output/lab4_bw.png")
print("Saved lab4_output.png")
# plt.show()
