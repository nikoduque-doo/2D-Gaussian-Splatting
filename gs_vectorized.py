import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from tqdm import tqdm

# --- Parameters ---
threshold = 0.3
truncate = 3
max_points = 500000
checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

# --- Load and normalize image ---
image = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0

# --- Convert to grayscale to select significant points ---
gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
gray = gray / 255.0
points = np.argwhere(gray > threshold)

if len(points) > max_points:
    points = points[np.random.choice(len(points), max_points, replace=False)]

height, width, _ = image.shape

# --- Gaussian splatting function ---
def splat_point(mu, color, sigma, output):
    eigvals = np.linalg.eigvalsh(sigma)
    radius = int(truncate * np.sqrt(np.max(eigvals)))

    x0, y0 = int(mu[0]), int(mu[1])
    x_min, x_max = max(0, x0 - radius), min(width - 1, x0 + radius)
    y_min, y_max = max(0, y0 - radius), min(height - 1, y0 + radius)

    X, Y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    positions = np.stack([X - mu[0], Y - mu[1]], axis=-1)

    inv_sigma = np.linalg.inv(sigma)
    M = np.einsum("...i,ij,...j->...", positions, inv_sigma, positions)
    G = np.exp(-0.5 * M)

    output[y_min:y_max+1, x_min:x_max+1, 0] += G * color[0]
    output[y_min:y_max+1, x_min:x_max+1, 1] += G * color[1]
    output[y_min:y_max+1, x_min:x_max+1, 2] += G * color[2]

# --- Covariance from scale and rotation ---
def build_sigma(scale_x, scale_y, theta_degrees):
    theta = np.radians(theta_degrees)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    S = np.diag([scale_x, scale_y])
    return R @ S @ S.T @ R.T

# --- Define fixed covariance families ---
covariance_configs = [
    (3, 3, 0),
    (4, 2, 0),
    (2, 4, 0),
    (3, 1, 45),
    (1, 3, -45)
]

results = []

with open("results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Config", "Scale X", "Scale Y", "Angle", "MSE", "MAE", "PSNR"])

    for idx, (sx, sy, angle) in enumerate(covariance_configs):
        sigma = build_sigma(sx, sy, angle)
        output = np.zeros((height, width, 3))

        print(f"Rendering config {idx+1}: scale=({sx}, {sy}), angle={angle}°")

        for (y, x) in tqdm(points, desc=f"  → Config {idx+1}/{len(covariance_configs)}", unit="pt"):
            color = image[y, x]
            splat_point((x, y), color, sigma, output)

        output = np.clip(output / output.max(), 0, 1)

        mse = np.mean((image - output) ** 2)
        mae = np.mean(np.abs(image - output))
        psnr = -10 * np.log10(mse) if mse > 0 else float("inf")

        results.append((sigma, mse))

        fname = f"checkpoints/config_{idx+1:02d}_mse_{mse:.6f}.png"
        cv2.imwrite(fname, (output * 255).astype(np.uint8)[:, :, ::-1])

        writer.writerow([idx + 1, sx, sy, angle, mse, mae, psnr])

# --- Summary ---
print("\nSummary of configurations:")
for i, (sigma, mse) in enumerate(results):
    print(f"  Config {i+1}: MSE={mse:.6f}\n{sigma}\n")
