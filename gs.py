import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Parámetros globales ---
truncate = 3  # truncar a 3 sigmas
threshold = 0.2

# --- 1. Cargar imagen y normalizar ---
image = cv2.imread("input.jpg", cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0

# --- 2. Convertir a gris para extraer puntos relevantes ---
gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
gray = gray / 255.0
points = np.argwhere(gray > threshold)

# --- 3. Crear buffers RGB ---
height, width, _ = image.shape
output_r = np.zeros((height, width))
output_g = np.zeros((height, width))
output_b = np.zeros((height, width))

# --- 4. Gaussiana 2D con covarianza ---
def gaussian_anisotropic(x, y, mu, inv_sigma):
    d = np.array([x - mu[0], y - mu[1]])
    return np.exp(-0.5 * d @ inv_sigma @ d)

# --- 5. Acumulación de splats con anisotropía ---
for y, x in points:
    r, g, b = image[y, x]
    mu = (x, y)

    # Ejemplo: matriz de covarianza anisotrópica para cada punto
    sigma = np.array([[4.0, 1.5], [1.5, 2.0]])  # forma y orientación
    inv_sigma = np.linalg.inv(sigma)

    # Calcular rango de evaluación
    radius = int(truncate * max(np.sqrt(np.linalg.eigvals(sigma))))
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < width and 0 <= yy < height:
                w = gaussian_anisotropic(xx, yy, mu, inv_sigma)
                output_r[yy, xx] += r * w
                output_g[yy, xx] += g * w
                output_b[yy, xx] += b * w

# --- 6. Normalización ---
output_rgb = np.stack([output_r, output_g, output_b], axis=-1)
output_rgb = np.clip(output_rgb / output_rgb.max(), 0, 1)

# --- 7. Visualización ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output_rgb)
plt.title("Splatting RGB con anisotropía")
plt.axis("off")

plt.tight_layout()
plt.show()
plt.show(block=True)
