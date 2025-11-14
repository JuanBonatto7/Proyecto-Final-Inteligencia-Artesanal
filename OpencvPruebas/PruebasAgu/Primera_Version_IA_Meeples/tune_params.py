#!/usr/bin/env python3
"""
Script para ajustar parámetros de detección de meeples
"""

from src.meeple_detector_cv import MeepleDetector
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def tune_circle_detection(image_path: str):
    """Permite ajustar parámetros de detección de círculos interactivamente"""

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ No se pudo cargar la imagen: {image_path}")
        return

    # Parámetros iniciales
    params = {
        'dp': 1.2,
        'minDist': 30,
        'param1': 50,
        'param2': 30,
        'minRadius': 10,
        'maxRadius': 50
    }

    print("🔧 AJUSTE DE PARÁMETROS DE DETECCIÓN DE CÍRCULOS")
    print("Presiona Enter para usar valor por defecto, o ingresa nuevo valor")
    print("-" * 50)

    for param_name in params:
        current_value = params[param_name]
        try:
            new_value = input(f"{param_name} (actual: {current_value}): ").strip()
            if new_value:
                if param_name in ['dp']:
                    params[param_name] = float(new_value)
                else:
                    params[param_name] = int(new_value)
        except ValueError:
            print(f"Valor inválido, manteniendo {current_value}")

    # Probar detección con parámetros ajustados
    detector = MeepleDetector()
    detector.circle_params = params

    print("\n🔍 Probando detección con parámetros ajustados...")

    # Detectar círculos
    circles = detector.detect_circles(image)

    print(f"Círculos detectados: {len(circles)}")

    # Visualizar
    vis_image = image.copy()
    for x, y, r in circles:
        cv2.circle(vis_image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(vis_image, (x, y), 2, (0, 0, 255), 3)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detección de círculos - {len(circles)} encontrados")
    plt.axis('off')
    plt.show()

    # Preguntar si guardar parámetros
    save = input("\n¿Guardar estos parámetros? (s/n): ").lower().strip()
    if save == 's':
        with open('circle_params.json', 'w') as f:
            import json
            json.dump(params, f, indent=2)
        print("✅ Parámetros guardados en: circle_params.json")

def tune_color_detection(image_path: str):
    """Permite ajustar rangos de color"""

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ No se pudo cargar la imagen: {image_path}")
        return

    print("🎨 AJUSTE DE RANGOS DE COLOR")
    print("Esto requiere ajuste manual del código para rangos HSV")

    # Convertir a HSV para mostrar
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Imagen Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(hsv[:, :, 0], cmap='hsv')
    plt.title("Canal H (Hue)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hsv[:, :, 1], cmap='gray')
    plt.title("Canal S (Saturation)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("💡 Para ajustar rangos de color, modifica los valores en MeepleDetector.color_ranges")

def main():
    parser = argparse.ArgumentParser(description='Ajustar parámetros de detección')
    parser.add_argument('image_path', help='Ruta a imagen de prueba')
    parser.add_argument('--mode', choices=['circles', 'colors'], default='circles',
                       help='Qué ajustar: círculos o colores')

    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Imagen no encontrada: {image_path}")
        return

    if args.mode == 'circles':
        tune_circle_detection(image_path)
    elif args.mode == 'colors':
        tune_color_detection(image_path)

if __name__ == "__main__":
    import argparse
    main()