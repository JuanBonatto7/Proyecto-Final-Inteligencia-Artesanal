#!/usr/bin/env python3
"""
Script de debug para el detector de meeples
"""

import cv2
import numpy as np
from pathlib import Path
from src.meeple_detector_cv import MeepleDetector

def debug_detection(image_path: str):
    """Debug detallado de la detección"""

    print(f"🔍 DEBUG: {image_path}")

    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print("❌ No se pudo cargar la imagen")
        return

    print(f"Dimensiones: {image.shape}")

    detector = MeepleDetector()

    # 1. Mostrar imagen original
    cv2.imshow("Original", image)
    cv2.waitKey(1000)

    # 2. Probar detección de círculos con diferentes parámetros
    print("\n⭕ Probando detección de círculos...")

    param_configs = [
        {'dp': 1.2, 'param2': 30, 'minRadius': 10, 'maxRadius': 50},
        {'dp': 1.5, 'param2': 25, 'minRadius': 5, 'maxRadius': 60},
        {'dp': 1.0, 'param2': 20, 'minRadius': 3, 'maxRadius': 80},
    ]

    for i, params in enumerate(param_configs):
        print(f"Config {i+1}: {params}")
        detector.circle_params.update(params)

        circles = detector.detect_circles(image)
        print(f"  Círculos encontrados: {len(circles)}")

        if circles:
            vis = image.copy()
            for x, y, r in circles:
                cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
                cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)

                # Analizar color del círculo
                color = detector.get_circle_color(image, (x, y, r))
                print(f"    Círculo en ({x}, {y}), radio {r}: {color}")

            cv2.imshow(f"Circulos Config {i+1}", vis)
            cv2.waitKey(2000)

    # 3. Análisis de color en puntos específicos
    print("\n🎨 Análisis de color en puntos...")

    # Si sabemos que hay un meeple en posición específica, analizar ese punto
    height, width = image.shape[:2]

    # Analizar el centro de cada región de la cuadrícula 3x3
    for pos in range(9):
        row = pos // 3
        col = pos % 3
        center_x = int((col + 0.5) * width / 3)
        center_y = int((row + 0.5) * height / 3)

        # Analizar un círculo pequeño en este punto
        test_radius = min(width, height) // 15
        color = detector.get_circle_color(image, (center_x, center_y, test_radius))
        print(f"  Posición {pos} (centro {center_x},{center_y}): {color}")

        # Mostrar punto
        vis = image.copy()
        cv2.circle(vis, (center_x, center_y), test_radius, (255, 0, 255), 2)
        cv2.putText(vis, f"Pos {pos}: {color}", (center_x-50, center_y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.imshow(f"Posicion {pos}", vis)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()

def main():
    import sys
    if len(sys.argv) != 2:
        print("Uso: python debug_detector.py <ruta_imagen>")
        return

    image_path = sys.argv[1]
    debug_detection(image_path)

if __name__ == "__main__":
    main()