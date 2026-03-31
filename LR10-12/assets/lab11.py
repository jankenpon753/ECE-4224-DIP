import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os

os.makedirs("./images/output", exist_ok=True)

# Step 1: Define boundary image A
A = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=bool,
)

# Step 2: Complement of A
Ac = ~A

# Step 3: Structuring Element (cross shape)
B = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

# Step 4: Seed point
# MATLAB coordinates (4,4) map to Python 0-based index [3, 3]
X_prev = np.zeros_like(A)
X_prev[3, 3] = 1

# Save initial seed state for visualization later
X0 = X_prev.copy()

# Step 5: Iteration (Morphological Region Filling Algorithm)
while True:
    # Dilate the previous region
    X_dilated = binary_dilation(X_prev, structure=B)

    # Intersect with the complement of the boundary
    X_new = X_dilated & Ac

    # Check for convergence (if the region stops growing)
    if np.array_equal(X_new, X_prev):
        break

    X_prev = X_new.copy()

Xk = X_new

# Step 6: Final result
# Union of original boundary and the filled region
Filled = A | Xk

# Step 7: Display Setup
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()


# Custom Plotting Function matching your MATLAB style
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
showMatrix(axes[0], A, "Original Boundary A")
showMatrix(axes[1], Ac, "Complement $A^c$")
showMatrix(axes[2], X0, "Initial Seed $X_0$")
showMatrix(axes[3], Xk, "Filled Region $X_k$")
showMatrix(axes[4], Filled, "Final Filled Image")
showMatrix(axes[5], B, "Structuring Element B")

plt.tight_layout()
plt.savefig("./images/output/lab11_fill.png")
print("Saved lab11_fill.png to output folder.")
# plt.show()
