#!/usr/bin/env python3
"""
Script rápido para probar el detector con diferentes configuraciones
"""

import cv2
import numpy as np
from pathlib import Path
from src.meeple_detector_cv import MeepleDetector

def quick_test(image_path: str):
    """Prueba rápida del detector"""

    print(f"🔍 Analizando: {image_path}")

    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print("❌ No se pudo cargar la imagen")
        return

    print(f"Dimensiones: {image.shape}")

    # Crear detector
    detector = MeepleDetector()

    # 1. Probar detección de borde
    print("\n🔲 Probando detección de borde...")
    border = detector.detect_tile_border(image)
    if border is not None:
        print("✅ Borde detectado")
        # Dibujar borde
        vis = image.copy()
        cv2.drawContours(vis, [border.astype(int)], -1, (0, 255, 0), 3)
        cv2.imshow("Borde Detectado", vis)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    else:
        print("❌ No se detectó borde")

    # 2. Probar detección de círculos con diferentes parámetros
    print("\n⭕ Probando detección de círculos...")

    # Parámetros a probar
    param_sets = [
        {'dp': 1.2, 'param2': 30, 'minRadius': 10, 'maxRadius': 50},  # Default
        {'dp': 1.5, 'param2': 25, 'minRadius': 5, 'maxRadius': 40},   # Más sensible
        {'dp': 1.0, 'param2': 35, 'minRadius': 15, 'maxRadius': 60},  # Menos sensible
    ]

    for i, params in enumerate(param_sets):
        print(f"  Config {i+1}: {params}")
        detector.circle_params.update(params)

        circles = detector.detect_circles(image)
        print(f"    Círculos encontrados: {len(circles)}")

        if circles:
            # Mostrar círculos
            vis = image.copy()
            for x, y, r in circles:
                cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
                cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)

            cv2.imshow(f"Circulos Config {i+1}", vis)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()

    # 3. Procesamiento completo
    print("\n🚀 Procesamiento completo...")
    result = detector.process_image(image_path)

    print("Resultado final:")
    print(f"  Loseta detectada: {'Sí' if result['tile_detected'] else 'No'}")
    print(f"  Meeples encontrados: {result['meeples_found']}")

    for meeple in result['meeples']:
        print(f"    {meeple['color']} en posición {meeple['position']}")

def main():
    # Probar con todas las imágenes disponibles
    tiles_dir = Path("tiles")
    if not tiles_dir.exists():
        print("❌ Directorio 'tiles' no encontrado")
        return

    image_files = list(tiles_dir.glob("*.png"))[:3]  # Solo las primeras 3 para no tardar tanto

    if not image_files:
        print("❌ No se encontraron imágenes PNG")
        return

    print(f"🎯 Probando con {len(image_files)} imágenes")

    for image_file in image_files:
        quick_test(str(image_file))
        input("\nPresiona Enter para continuar con la siguiente imagen...")

if __name__ == "__main__":
    main()