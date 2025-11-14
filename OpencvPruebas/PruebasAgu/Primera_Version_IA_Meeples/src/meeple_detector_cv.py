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
        # Parámetros para detección de círculos - ajustados para meeples perfectamente circulares
        self.circle_params = {
            'dp': 1.2,
            'minDist': 100,  # Mayor distancia - solo 1 meeple por imagen
            'param1': 100,   # Umbral de Canny - menos restrictivo
            'param2': 30,    # Umbral de acumulador - menos restrictivo
            'minRadius': 10, # Radio mínimo más pequeño
            'maxRadius': 80  # Radio máximo más grande
        }

        # Rangos de color para meeples azules y negros (valores exactos proporcionados)
        # Azul: HSV(212, 64%, 62%) -> H:106, S:163, V:158
        # Negro: HSV(240, 10%, 8%) -> H:120, S:26, V:20
        self.color_ranges = {
            'blue': {
                'lower': np.array([95, 140, 120]),   # Rango estrecho alrededor del azul exacto
                'upper': np.array([115, 180, 190])
            },
            'black': {
                'lower': np.array([0, 0, 0]),        # Negro - rango amplio pero con umbral bajo
                'upper': np.array([179, 50, 50])     # Valor máximo bajo para negro
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

        # Lógica de clasificación usando rangos precisos basados en valores reales
        # Azul: HSV(212, 64%, 62%) - rangos estrechos para mayor precisión
        blue_lower = self.color_ranges['blue']['lower']
        blue_upper = self.color_ranges['blue']['upper']

        # Negro: HSV(240, 10%, 8%) - rangos para negro oscuro
        black_lower = self.color_ranges['black']['lower']
        black_upper = self.color_ranges['black']['upper']

        # Crear máscaras para cada color
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        black_mask = cv2.inRange(hsv, black_lower, black_upper)

        # Contar píxeles de cada color
        blue_pixels = cv2.countNonZero(blue_mask)
        black_pixels = cv2.countNonZero(black_mask)

        # Si hay suficientes píxeles azules, clasificar como azul
        if blue_pixels > len(mask[mask > 0]) * 0.3:  # Al menos 30% de píxeles azules
            return 'blue'

        # Si hay suficientes píxeles negros, clasificar como negro
        if black_pixels > len(mask[mask > 0]) * 0.3:  # Al menos 30% de píxeles negros
            return 'black'

        # Fallback: usar estadísticas de color si la máscara no funciona
        if ((95 <= mean_hue <= 115) and (140 <= mean_sat <= 180) and (120 <= mean_val <= 190)):
            return 'blue'

        if mean_val < 50 and mean_sat < 50:
            return 'black'

        # Si no cumple criterios claros, usar análisis BGR mejorado
        b, g, r = mean_bgr

        # Azul: componente azul dominante, con cierto brillo
        if b > g + 20 and b > r + 20 and b > 60:
            return 'blue'

        # Negro: todos los componentes bajos (oscuro)
        if b < 70 and g < 70 and r < 70:
            return 'black'

        return 'unknown'

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

        # Convertir a HSV para usar rangos precisos
        hsv_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        # Usar los mismos rangos que en get_circle_color
        blue_mask = cv2.inRange(hsv_masked, self.color_ranges['blue']['lower'], self.color_ranges['blue']['upper'])
        black_mask = cv2.inRange(hsv_masked, self.color_ranges['black']['lower'], self.color_ranges['black']['upper'])

        blue_pixels = cv2.countNonZero(blue_mask)
        black_pixels = cv2.countNonZero(black_mask)

        total_pixels = cv2.countNonZero(mask)

        if blue_pixels > total_pixels * 0.2:  # 20% threshold para clasificación forzada
            return 'blue'
        elif black_pixels > total_pixels * 0.2:
            return 'black'

        # Fallback BGR si HSV no funciona
        b, g, r = mean_color
        if b > g + 15 and b > r + 15 and b > 80:  # Azul en BGR
            return 'blue'
        elif b < 60 and g < 60 and r < 60:  # Negro en BGR
            return 'black'

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

        # Procesar cada círculo detectado - cada imagen tiene máximo 1 meeple
        meeples = []

        # Filtrar círculos por tamaño (meeples son perfectamente circulares y de tamaño consistente)
        valid_circles = []
        for circle in circles:
            x, y, r = circle
            # Meeples tienen radio entre 10-80 píxeles aproximadamente
            if 10 <= r <= 80:
                valid_circles.append(circle)

        # Si hay múltiples círculos, elegir el mejor candidato (más circular, mejor color)
        if valid_circles:
            # Calcular "calidad" de cada círculo basado en circularidad y color
            best_circle = None
            best_score = -1

            for circle in valid_circles:
                x, y, r = circle

                # Clasificar color
                color = self.get_circle_color(image, circle)
                if color == 'unknown':
                    color = self._force_color_classification(image, circle)

                # Puntaje basado en color (azul o negro son mejores que unknown)
                color_score = 2 if color in ['blue', 'black'] else 0

                # Puntaje basado en tamaño (meeples de tamaño mediano son mejores)
                size_score = 1 if 20 <= r <= 45 else 0

                total_score = color_score + size_score

                if total_score > best_score:
                    best_score = total_score
                    best_circle = circle

            # Usar el mejor círculo encontrado
            if best_circle:
                x, y, r = best_circle
                position = self.determine_position(best_circle, regions)

                # Clasificar color final
                color = self.get_circle_color(image, best_circle)
                if color == 'unknown':
                    color = self._force_color_classification(image, best_circle)

                # Solo aceptar azules y negros
                if color in ['blue', 'black']:
                    meeples.append({
                        'color': color,
                        'position': position if position != -1 else 4,  # Centro por defecto
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
            elif color == 'black':
                draw_color = (0, 0, 0)   # Negro
            else:
                draw_color = (128, 128, 128)  # Gris para unknown

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