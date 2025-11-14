#!/usr/bin/env python3
"""
Script para probar el detector de meeples con visualización
"""

import argparse
from pathlib import Path
from src.meeple_detector_cv import MeepleDetector

def main():
    parser = argparse.ArgumentParser(description='Detectar meeples en imágenes usando OpenCV')
    parser.add_argument('image_path', help='Ruta a la imagen a procesar')
    parser.add_argument('--save', '-s', help='Guardar visualización en archivo')

    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Imagen no encontrada: {image_path}")
        return

    # Crear detector
    detector = MeepleDetector()

    # Procesar imagen
    print(f"Procesando: {image_path}")
    result = detector.process_image(image_path)

    if 'error' in result:
        print(result['error'])
        return

    # Mostrar resultados
    print("\n📊 RESULTADOS:")
    print(f"Loseta detectada: {'Sí' if result['tile_detected'] else 'No'}")
    print(f"Meeples encontrados: {result['meeples_found']}")

    for i, meeple in enumerate(result['meeples'], 1):
        print(f"  {i}. Meeple {meeple['color']} en posición {meeple['position']}")
        x, y, r = meeple['circle']
        print(f"     Centro: ({x}, {y}), Radio: {r}")

    # Visualizar
    save_path = args.save if args.save else None
    detector.visualize_detection(image_path, save_path)

if __name__ == "__main__":
    main()