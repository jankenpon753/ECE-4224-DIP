import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_hit_or_miss

print("\n--- Experiment 9: Hit-or-Miss Transform ---")

# 1. Define image
img_full = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

# 2. Define structuring elements
# SE1: pattern to find (2x2 square of 1s)
SE1 = np.array([[1, 1], [1, 1]])

# SE2: background around the pattern
SE2 = np.array([[0, 0], [0, 0]])

# 3. Apply Hit-or-Miss
hmt_result = binary_hit_or_miss(img_full, structure1=SE1, structure2=SE2).astype(int)

print("Hit-or-Miss Result (1 indicates pattern found):\n", hmt_result)

# 4. Visualize
output_dir = "./images/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "lab9.png")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_full, cmap="gray")
plt.title("Original Image")
# Add gridlines
ax1 = plt.gca()
ax1.set_xticks(np.arange(-0.5, img_full.shape[1], 1), minor=True)
ax1.set_yticks(np.arange(-0.5, img_full.shape[0], 1), minor=True)
ax1.grid(which="minor", color="red", linestyle="-", linewidth=0.5)
plt.subplot(1, 2, 2)
plt.imshow(hmt_result, cmap="gray")
plt.title("Hit-or-Miss Result")
# Add gridlines
ax2 = plt.gca()
ax2.set_xticks(np.arange(-0.5, hmt_result.shape[1], 1), minor=True)
ax2.set_yticks(np.arange(-0.5, hmt_result.shape[0], 1), minor=True)
ax2.grid(which="minor", color="red", linestyle="-", linewidth=0.5)
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Image saved to {output_path}")
plt.show()
