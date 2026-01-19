import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")

# --- PART 1: Basic Operations ---
# 1. Read Image
a_bgr = cv2.imread("./images/Lenna.png")
if a_bgr is None:
    raise FileNotFoundError(
        "Image not found. Please ensure 'Lenna.png' is in the working directory."
    )

a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)

# 2. Resize # Resize to 50x50 pixels
b = cv2.resize(a, (50, 50))

# 3. Image Info
print("Image Info:")
print(f"Shape: {a.shape}")
print(f"Data Type: {a.dtype}")

# 4. Convert to Grayscale
c = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

# 5. Convert to Binary
_, d = cv2.threshold(c, 127, 255, cv2.THRESH_BINARY)

# --- PART 2: Manipulations ---
# 1. Merge Image (Grayscale + Binary)
m = np.hstack((c, d))

# 2. Channel Manipulation (Blue Channel Only)
a_mod = a.copy()
a_mod[:, :, 0] = 0  # Set Red channel to 0
a_mod[:, :, 1] = 0  # Set Green channel to 0

# 3. Flip and Rotate
n = cv2.flip(a, 0)  # Vertical
o = cv2.flip(a, 1)  # Horizontal
p = cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)

# 4. Change Color Intensity
red_factor = 0.2
green_factor = 1.8
blue_factor = 0.5

# Scale individual channels
red_channel = np.clip(a[:, :, 0] * red_factor, 0, 255).astype(np.uint8)
green_channel = np.clip(a[:, :, 1] * green_factor, 0, 255).astype(np.uint8)
blue_channel = np.clip(a[:, :, 2] * blue_factor, 0, 255).astype(np.uint8)

# Recombine channels
modified_img = cv2.merge((red_channel, green_channel, blue_channel))

# --- SUMMARY DISPLAY (All 10 Results) ---
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Basic Operations Complete Summary", fontsize=16)

# Row 1: Basic Conversions
# 1. Original
axes[0, 0].imshow(a)
axes[0, 0].set_title("1. Original")
axes[0, 0].axis("off")
# 2. Resized
axes[0, 1].imshow(b)
axes[0, 1].set_title("2. Resized (50x50)")
axes[0, 1].axis("off")
# 3. Grayscale (needs cmap='gray')
axes[0, 2].imshow(c, cmap="gray")
axes[0, 2].set_title("3. Grayscale")
axes[0, 2].axis("off")
# 4. Binary (needs cmap='gray')
axes[0, 3].imshow(d, cmap="gray")
axes[0, 3].set_title("4. Binary")
axes[0, 3].axis("off")
# 5. Merged (needs cmap='gray')
axes[0, 4].imshow(m, cmap="gray")
axes[0, 4].set_title("5. Merged (Gray+Bin)")
axes[0, 4].axis("off")

# Row 2: Manipulations
# 6. Blue Channel Only
axes[1, 0].imshow(a_mod)
axes[1, 0].set_title("6. Blue Channel Only")
axes[1, 0].axis("off")
# 7. Flip Vertical
axes[1, 1].imshow(n)
axes[1, 1].set_title("7. Flip Vertical")
axes[1, 1].axis("off")
# 8. Flip Horizontal
axes[1, 2].imshow(o)
axes[1, 2].set_title("8. Flip Horizontal")
axes[1, 2].axis("off")
# 9. Rotated
axes[1, 3].imshow(p)
axes[1, 3].set_title("9. Rotated 90°")
axes[1, 3].axis("off")
# 10. Color Modified
axes[1, 4].imshow(modified_img)
axes[1, 4].set_title("10. Color Modified")
axes[1, 4].axis("off")

plt.tight_layout()
plt.show()
