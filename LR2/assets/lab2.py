from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("./images/output", exist_ok=True)

# Path to image
img_path = "./images/Lenna.png"


# Task 1: White Cross
# Load Lenna as RGB and convert to numpy array
img = Image.open(img_path).convert("RGB")
arr = np.array(img)

# Compute image center (row = cy, column = cx)
h, w, _ = arr.shape
cy, cx = h // 2, w // 2

# Draw a white horizontal line through the center row
arr[cy, :, :] = 255
# Draw a white vertical line through the center column
arr[:, cx, :] = 255

# Convert back to PIL image, display, and save
modified_img = Image.fromarray(arr)
# modified_img.show()
modified_img.save("./images/output/task1_cross.png")
print("Task 1: Saved './images/output/task1_cross.png'")

# Task 2: Color Bands (Red Vertical, Blue Horizontal)
# Reload Lenna image and convert to array
img_band = Image.open(img_path).convert("RGB")
arr_band = np.array(img_band)

# Define band thickness
band_half = 50  # total width/height = 100 pixels

# Apply vertical red-only band (centered)
# Logic: Keep Red channel, set Green (index 1) and Blue (index 2) to 0
arr_band[:, cx - band_half : cx + band_half, 1] = 0  # zero green
arr_band[:, cx - band_half : cx + band_half, 2] = 0  # zero blue

# Apply horizontal blue-only band (centered)
# Logic: Keep Blue channel, set Red (index 0) and Green (index 1) to 0
arr_band[cy - band_half : cy + band_half, :, 0] = 0  # zero red
arr_band[cy - band_half : cy + band_half, :, 1] = 0  # zero green

# Convert back to image, display, and save
band_img = Image.fromarray(arr_band)
# band_img.show()
band_img.save("./images/output/task2_bands.png")
print("Task 2: Saved './images/output/task2_bands.png'")

# Task 3: Manual Histogram Loop
# Load Lenna image and convert to grayscale
gray_img_manual = Image.open(img_path).convert("L")
gray_arr_manual = np.array(gray_img_manual)

# Manually compute histogram using a loop
hist_manual = np.zeros(256, dtype=int)
for pixel_value in gray_arr_manual.ravel():
    hist_manual[pixel_value] += 1

# Plot the histogram
plt.figure(figsize=(6, 4))
plt.bar(range(256), hist_manual, width=1.0, color="gray")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.title("Lenna grayscale histogram (manual loop)")
plt.tight_layout()

# Save the plot
plt.savefig("./images/output/task3_histogram.png")
print("Task 3: Saved './images/output/task3_histogram.png'")
# plt.show()
