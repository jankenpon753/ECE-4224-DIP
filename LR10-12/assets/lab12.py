import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os

os.makedirs("./images/output", exist_ok=True)

# Step 1: Define binary image A
A = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    ],
    dtype=bool,
)

# Step 2: Structuring Element (8-connected block)
# Note: The MATLAB comment says "4-connected" but the matrix is a 3x3 of all 1s (which is 8-connected).
B = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)

# Step 3: Select a seed point from a component
# MATLAB coordinates (3,4) map to Python 0-based index [1,1]
X_prev = np.zeros_like(A)
X_prev[3, 8] = 1

# Save initial seed state for visualization
X0 = X_prev.copy()

# Step 4: Iterative extraction (Connected Component Algorithm)
while True:
    # Dilate the previous region
    X_dilated = binary_dilation(X_prev, structure=B)

    # Intersect with the original image to constrain growth to foreground pixels
    X_new = X_dilated & A

    # Check for convergence
    if np.array_equal(X_new, X_prev):
        break

    X_prev = X_new.copy()

Xk = X_new  # Final connected component

# -------- Display Setup --------
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()


# Custom Plotting Function matching your style
def showMatrix(ax, mat, titleText):
    ax.imshow(mat, cmap="gray", vmin=0, vmax=1)
    ax.set_title(titleText, fontsize=14)

    # Set up grid lines on pixel borders
    h, w = mat.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    # Hide major ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add red numbers to the center of each cell
    for i in range(h):
        for j in range(w):
            val = int(mat[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="red",
                fontweight="bold",
                fontsize=12,
            )


# Plot all stages
showMatrix(axes[0], A, "Original Image A")
showMatrix(axes[1], X0, "Initial Seed $X_0$")
showMatrix(axes[2], Xk, "Extracted Connected Component $X_k$")
showMatrix(axes[3], B, "Structuring Element B")

plt.tight_layout()
plt.savefig("./images/output/lab12_connected.png")
print("Saved lab12_connected.png to output folder.")
# plt.show()
