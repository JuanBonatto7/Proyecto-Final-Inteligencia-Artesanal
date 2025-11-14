#!/usr/bin/env python3
"""
Detector de Meeples usando OpenCV - Approach basado en visión computacional
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

class MeepleDetector:
    """Detector de meeples usando visión computacional"""

    def __init__(self):
        # Parámetros para detección de círculos
        self.circle_params = {
            'dp': 1.2,
            'minDist': 50,  # Mayor distancia mínima entre círculos
            'param1': 100,  # Umbral de Canny más alto
            'param2': 40,   # Umbral de acumulador más alto (más selectivo)
            'minRadius': 20, # Radio mínimo mayor
            'maxRadius': 80  # Radio máximo
        }

        # Rangos de color para meeples (ajustados para imágenes reales)
        self.color_ranges = {
            'blue': {
                'lower': np.array([80, 30, 30]),   # Azul original (para imágenes simuladas)
                'upper': np.array([140, 255, 255])
            },
            'red_orange': {  # Nuevo rango para meeples rojizos/anaranjados en fotos reales
                'lower': np.array([0, 50, 100]),   # Hue bajo (rojo-anaranjado), saturación media, valor alto
                'upper': np.array([25, 255, 255])
            },
            'black': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 80])  # Más permisivo para negro
            }
        }

    def detect_tile_border(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detecta el borde de la loseta usando contornos
        Retorna los puntos del contorno principal (loseta)
        """
        # Convertir a gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detectar bordes con Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Encontrar el contorno más grande (asumiendo que es la loseta)
        largest_contour = max(contours, key=cv2.contourArea)

        # Aproximar el contorno a un rectángulo
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Si tiene 4 puntos, es probablemente un rectángulo (loseta)
        if len(approx) == 4:
            # Ordenar puntos: top-left, top-right, bottom-right, bottom-left
            points = approx.reshape(4, 2)
            rect = self._order_points(points)
            return rect

        return None

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Ordena puntos de un rectángulo en orden clockwise desde top-left"""
        # Ordenar por suma (x+y) - top-left tendrá la menor suma
        # bottom-right tendrá la mayor suma
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)

        top_left = pts[np.argmin(sum_pts)]
        bottom_right = pts[np.argmax(sum_pts)]
        top_right = pts[np.argmin(diff_pts)]
        bottom_left = pts[np.argmax(diff_pts)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def divide_into_9_regions_simple(self, image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Divide la imagen en 9 regiones simples (sin detección de bordes)
        """
        h, w = image_shape[:2]
        regions = []
        for i in range(3):
            for j in range(3):
                x = int(j * w / 3)
                y = int(i * h / 3)
                w_region = int(w / 3)
                h_region = int(h / 3)
                regions.append([x, y, w_region, h_region])
        return regions

    def detect_circles(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detecta círculos en la imagen usando Hough Circle Transform
        Retorna lista de (x, y, radius)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detectar círculos
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_params['dp'],
            minDist=self.circle_params['minDist'],
            param1=self.circle_params['param1'],
            param2=self.circle_params['param2'],
            minRadius=self.circle_params['minRadius'],
            maxRadius=self.circle_params['maxRadius']
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            return [(x, y, r) for x, y, r in circles]

        return []

    def get_circle_color(self, image: np.ndarray, circle: Tuple[int, int, int]) -> str:
        """
        Determina el color de un círculo (meeple)
        Retorna 'blue', 'black', o 'unknown'
        """
        x, y, r = circle

        # Crear máscara circular
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Extraer región del círculo
        masked = cv2.bitwise_and(image, image, mask=mask)

        # Convertir a HSV
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        # Calcular histogramas
        hist_h = cv2.calcHist([hsv], [0], mask, [180], [0, 180])  # Hue
        hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])  # Saturation
        hist_v = cv2.calcHist([hsv], [2], mask, [256], [0, 256])  # Value

        # Estadísticas básicas
        mean_hue = np.average(range(180), weights=hist_h.flatten())
        mean_sat = np.average(range(256), weights=hist_s.flatten())
        mean_val = np.average(range(256), weights=hist_v.flatten())

        # Convertir a BGR para análisis adicional
        bgr_region = masked[mask > 0]
        if len(bgr_region) == 0:
            return 'unknown'

        mean_bgr = np.mean(bgr_region, axis=0)

        # Lógica de clasificación mejorada
        # Azul: Hue entre 80-140, buena saturación, buen valor
        if (80 <= mean_hue <= 140 and mean_sat > 30 and mean_val > 40):
            return 'blue'

        # Rojo/Anaranjado: Hue bajo (0-25), buena saturación, buen valor (para fotos reales)
        if (0 <= mean_hue <= 25 and mean_sat > 50 and mean_val > 100):
            return 'red_orange'

        # Negro: Bajo valor, baja saturación
        if mean_val < 80 and mean_sat < 50:
            return 'black'

        # Si no cumple criterios claros, usar análisis BGR
        b, g, r = mean_bgr
        if b > g + 20 and b > r + 20:  # Azul dominante
            return 'blue'
        elif r > b + 20 and r > g + 20:  # Rojo dominante (para fotos reales)
            return 'red_orange'
        elif b < 60 and g < 60 and r < 60:  # Todo oscuro
            return 'black'

        return 'unknown'

    def _force_color_classification(self, image: np.ndarray, circle: Tuple[int, int, int]) -> str:
        """
        Clasificación forzada de color cuando la clasificación normal falla
        """
        x, y, r = circle

        # Crear máscara circular
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Extraer colores promedio
        masked = cv2.bitwise_and(image, image, mask=mask)
        mean_color = cv2.mean(masked, mask=mask)[:3]  # BGR

        b, g, r = mean_color

        # Lógica simple: si azul es el componente dominante = azul, si rojo es dominante = rojo, si todo es oscuro = negro
        if b > g + 30 and b > r + 30:
            return 'blue'
        elif r > b + 30 and r > g + 30:
            return 'red_orange'
        elif b < 80 and g < 80 and r < 80:
            return 'black'
        else:
            return 'unknown'

    def determine_position(self, circle: Tuple[int, int, int], regions: List[np.ndarray]) -> int:
        """
        Determina en qué posición (0-8) está el círculo
        """
        x, y, r = circle

        for i, region in enumerate(regions):
            rx, ry, rw, rh = region

            # Verificar si el centro del círculo está dentro de la región
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return i

        return -1  # No encontrado en ninguna región

    def process_image(self, image_path: str) -> Dict:
        """
        Procesa una imagen completa y retorna información sobre meeples detectados
        """
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': f'No se pudo cargar la imagen: {image_path}'}

        # Detectar borde de la loseta
        tile_border = self.detect_tile_border(image)

        # Dividir en 9 regiones (usar simple por ahora)
        regions = self.divide_into_9_regions_simple(image.shape)

        # Detectar círculos
        circles = self.detect_circles(image)

        # Procesar cada círculo detectado
        meeples = []

        # Agrupar círculos por posición aproximada
        position_candidates = {i: [] for i in range(9)}

        for circle in circles:
            x, y, r = circle
            position = self.determine_position(circle, regions)
            if position != -1 and 10 <= r <= 100:  # Radio razonable
                position_candidates[position].append(circle)

        # Para cada posición, tomar el mejor círculo candidato
        for position, candidates in position_candidates.items():
            if candidates:
                # Tomar el círculo más grande (más probable de ser un meeple)
                best_circle = max(candidates, key=lambda c: c[2])

                # Intentar clasificar color
                color = self.get_circle_color(image, best_circle)
                if color == 'unknown':
                    color = self._force_color_classification(image, best_circle)

                # DEBUG: Ser más permisivo - incluir también unknown por ahora
                if color in ['blue', 'black', 'red_orange', 'unknown']:
                    meeples.append({
                        'color': color if color != 'unknown' else 'unknown_test',
                        'position': position,
                        'circle': best_circle
                    })

        result = {
            'image_path': str(image_path),
            'tile_detected': tile_border is not None,
            'meeples_found': len(meeples),
            'meeples': meeples,
            'regions': regions
        }

        return result

    def visualize_detection(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualiza la detección de meeples en una imagen
        """
        result = self.process_image(image_path)

        if 'error' in result:
            print(result['error'])
            return

        # Cargar imagen
        image = cv2.imread(str(image_path))
        vis_image = image.copy()

        # Dibujar regiones
        for i, region in enumerate(result['regions']):
            x, y, w, h = region
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 255, 255), 1)

            # Etiqueta de región
            cv2.putText(vis_image, str(i), (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Dibujar meeples detectados
        for meeple in result['meeples']:
            x, y, r = meeple['circle']
            color = meeple['color']
            position = meeple['position']

            # Color para dibujar
            if color == 'blue':
                draw_color = (255, 0, 0)  # Azul
            elif color == 'red_orange':
                draw_color = (0, 0, 255)  # Rojo
            else:  # black
                draw_color = (0, 0, 0)   # Negro

            # Dibujar círculo
            cv2.circle(vis_image, (x, y), r, draw_color, 2)

            # Etiqueta
            label = f"{color[0].upper()}{position}"
            cv2.putText(vis_image, label, (x-10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

        # Mostrar resultado
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detección - {result['meeples_found']} meeples encontrados")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualización guardada en: {save_path}")
        else:
            plt.show()

def main():
    """Función principal para probar el detector"""
    detector = MeepleDetector()

    # Procesar todas las imágenes en tiles/
    tiles_dir = Path("tiles")
    if not tiles_dir.exists():
        print("❌ Directorio 'tiles' no encontrado")
        return

    results = []
    for image_file in tiles_dir.glob("*.jpg"):
        print(f"Procesando: {image_file.name}")
        result = detector.process_image(image_file)
        results.append(result)

        # Mostrar resumen
        if 'error' not in result:
            print(f"  Meeples encontrados: {result['meeples_found']}")
            for meeple in result['meeples']:
                print(f"    {meeple['color']} en posición {meeple['position']}")

    # Guardar resultados
    with open('detection_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Procesadas {len(results)} imágenes")
    print("Resultados guardados en: detection_results.json")

if __name__ == "__main__":
    main()