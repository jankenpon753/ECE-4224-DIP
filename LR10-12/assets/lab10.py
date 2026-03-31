import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
import os

os.makedirs("./images/output", exist_ok=True)

# Step 1: Define the binary image A
A = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=bool,
)

# Step 2 & 3: Define Structuring Element B (3x3 ones)
B = np.ones((3, 3), dtype=bool)

# Step 4: Perform erosion
C = binary_erosion(A, structure=B)

# Step 5: Boundary extraction
# Subtracting eroded image from original
boundary = A.astype(int) - C.astype(int)

# Step 6: Display results
fig, axes = plt.subplots(1, 3, figsize=(17, 5))


def plot_matrix(ax, matrix, title):
    ax.imshow(matrix, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    # Add grid lines for clarity
    h, w = matrix.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    # Annotate values
    for i in range(h):
        for j in range(w):
            val = int(matrix[i, j])
            color = "white" if val == 0 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)


plot_matrix(axes[0], A, "Original Image A")
plot_matrix(axes[1], C, r"Eroded Image (A $\ominus$ B)")
plot_matrix(axes[2], boundary, r"Boundary = A - (A $\ominus$ B)")

plt.tight_layout()
plt.savefig("./images/output/lab10_boundary.png")
print("Saved lab10_boundary.png to output folder.")
# plt.show()
