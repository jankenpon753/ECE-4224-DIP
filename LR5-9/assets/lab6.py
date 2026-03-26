import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# 1. Given points
r1, r2 = 70, 180
s1, s2 = 40, 220
L_max = 255

# 2. Calculate slopes (Alpha, Beta, Gamma)
alpha = s1 / r1
beta = (s2 - s1) / (r2 - r1)
gamma = (L_max - s2) / (L_max - r2)

print("\n--- Experiment 6: Contrast Stretching ---")
print(f"Alpha (Dark): {alpha:.2f}")
print(f"Beta (Mid): {beta:.2f}")
print(f"Gamma (Bright): {gamma:.2f}")

# Load Lenna image
img_exp6 = np.array(Image.open("./images/Lenna.png").convert("L"), dtype=np.uint8)
out_exp6 = np.zeros_like(img_exp6, dtype=np.float32)

# 3. Apply piecewise stretching
for i in range(img_exp6.shape[0]):
    for j in range(img_exp6.shape[1]):
        r = img_exp6[i, j]
        if r < r1:
            out_exp6[i, j] = alpha * r
        elif r1 <= r <= r2:
            out_exp6[i, j] = beta * (r - r1) + s1
        else:
            out_exp6[i, j] = gamma * (r - r2) + s2

print("Original min/max:", img_exp6.min(), img_exp6.max())
print("Stretched min/max:", out_exp6.min(), out_exp6.max())

# Save the image
output_dir = "./images/output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "lab6.png")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_exp6, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(np.round(out_exp6).astype(np.uint8), cmap="gray")
plt.title("Stretched Image")
plt.axis("off")
plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Image saved to {output_path}")
plt.show()
