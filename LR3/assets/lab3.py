import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.colors import ListedColormap
import os

os.makedirs("./images/output", exist_ok=True)

# 1. Define Image Matrix (4x4)
image = np.array([[1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])

# Define Start and End (User requested 1,1 to 4,4 -> Python 0,0 to 3,3)
START = (0, 0)
END = (3, 3)


# Helper: Check if bounds are valid
def is_valid(x, y, shape):
    return 0 <= x < shape[0] and 0 <= y < shape[1]


# Helper: Get value safely
def get_val(img, x, y):
    if is_valid(x, y, img.shape):
        return img[x, y]
    return 0


# 2. Neighbor Finding Logic
def get_neighbors(p, img, mode):
    x, y = p
    neighbors = []
    rows, cols = img.shape

    # Potential moves # N4: (dx, dy)
    moves_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Diagonals: (dx, dy)
    moves_diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # --- 4-Connectivity ---
    if mode == "4":
        for dx, dy in moves_4:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, img.shape) and img[nx, ny] == 1:
                neighbors.append((nx, ny))

    # --- 8-Connectivity ---
    elif mode == "8":
        # Add all 4-neighbors and Diagonal neighbors
        for dx, dy in moves_4 + moves_diag:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, img.shape) and img[nx, ny] == 1:
                neighbors.append((nx, ny))

    # --- m-Connectivity ---
    elif mode == "m":
        # Condition 1: q is in N4(p)
        for dx, dy in moves_4:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, img.shape) and img[nx, ny] == 1:
                neighbors.append((nx, ny))

        # Condition 2: q is in Nd(p) AND intersection of N4 is empty
        for dx, dy in moves_diag:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny, img.shape) and img[nx, ny] == 1:
                # Check intersection neighbors (the two shared 4-neighbors)
                # For a diagonal move (dx, dy), the shared neighbors are (x+dx, y) and (x, y+dy)
                n4_1 = get_val(img, x + dx, y)
                n4_2 = get_val(img, x, y + dy)

                # Intersection must be empty (both 0)
                if n4_1 == 0 and n4_2 == 0:
                    neighbors.append((nx, ny))

    return neighbors


# 3. BFS Algorithm for Shortest Path
def bfs_shortest_path(img, start, end, mode):
    queue = deque([[start]])
    visited = set([start])

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == end:
            return path

        for neighbor in get_neighbors(node, img, mode):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None


# 4. Execution & Visualization
connectivities = ["4", "8", "m"]
results = {}

plt.figure(figsize=(12, 5))

for i, mode in enumerate(connectivities):
    path = bfs_shortest_path(image, START, END, mode)
    results[mode] = path

    # Visualization
    ax = plt.subplot(1, 3, i + 1)

    # Create visualization matrix: 0=Black, 1=White, 2=Path(Red)
    vis_img = np.zeros_like(image, dtype=float)
    vis_img[image == 1] = 1.0  # Foreground
    vis_img[image == 0] = 0.0  # Background

    path_len_str = "No Path"
    if path:
        path_len_str = f"Len: {len(path)} pixels"
        # Mark path
        for px, py in path:
            vis_img[px, py] = 0.5  # Grey/Red placeholder

    # Custom plotting overlaying the path manually to ensure it's visible
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)

    if path:
        # Unzip path into x and y lists for plotting lines
        ys, xs = zip(*path)  # x is row (y-axis in plot), y is col (x-axis in plot)
        ax.plot(
            xs, ys, color="red", linewidth=3, marker="o", markersize=8, label="Path"
        )

    ax.set_title(f"{mode}-Connectivity\n{path_len_str}")
    ax.invert_yaxis()  # Match matrix coordinates
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.grid(True, color="gray", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("./images/output/lab3_pathfinding.png")
# plt.show()

# 5. Text Output
print(f"Start: {(START[0]+1, START[1]+1)}")
print(f"End:   {(END[0]+1, END[1]+1)}\n")

for mode in connectivities:
    path = results[mode]
    print(f"--- {mode}-Connectivity ---")
    if path:
        # Convert to 1-based indexing for display
        path_1based = [(x + 1, y + 1) for x, y in path]
        print(f"Path Found: {path_1based}")
        print(f"Pixel Count: {len(path)}")
    else:
        print("Path Not Found")
    print("")
