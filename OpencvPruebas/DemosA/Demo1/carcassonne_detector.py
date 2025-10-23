import cv2
import numpy as np

# --- Cargar imagen ---
image = cv2.imread('tablero.jpg')
if image is None:
    raise FileNotFoundError(f"No se encontró la imagen en...")

# Redimensionar para acelerar
scale = 0.7
image = cv2.resize(image, None, fx=scale, fy=scale)
debug = image.copy()

# --- Preprocesamiento ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Detección de bordes ---
edges = cv2.Canny(blur, 40, 120)
# Unir líneas interrumpidas
kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)
edges = cv2.erode(edges, kernel, iterations=1)

# --- Buscar contornos ---
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tiles = []
for cnt in contornos:
    # Aproximar contorno a un polígono
    epsilon = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Debe tener 4 lados
    if len(approx) == 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        ratio = w / float(h)
        area = cv2.contourArea(cnt)

        # Filtros por tamaño y forma cuadrada
        if 0.85 < ratio < 1.15 and area > 1000:
            tiles.append((x, y, w, h))
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(f"Locetas detectadas: {len(tiles)}")

cv2.imwrite("debug_detectadas.jpg", debug)
print("Imagen 'debug_detectadas.jpg' generada con las locetas detectadas.")
