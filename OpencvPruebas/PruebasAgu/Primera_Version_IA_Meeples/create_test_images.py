#!/usr/bin/env python3
"""
Script para crear imágenes de prueba con meeples simulados
"""

import cv2
import numpy as np
from pathlib import Path
import random

def create_test_image_with_meeple(base_image_path: str, output_path: str,
                                meeple_color: str = 'blue', position: int = 4):
    """
    Crea una imagen de prueba agregando un meeple simulado a una loseta base
    """

    # Cargar imagen base
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        print(f"❌ No se pudo cargar: {base_image_path}")
        return False

    height, width = base_image.shape[:2]

    # Calcular posición del meeple basada en la cuadrícula 3x3
    row = position // 3
    col = position % 3

    # Centro de cada región
    center_x = int((col + 0.5) * width / 3)
    center_y = int((row + 0.5) * height / 3)

    # Radio del meeple (ajustable)
    radius = min(width, height) // 12  # Aproximadamente 1/12 del tamaño de la loseta

    # Color del meeple
    if meeple_color == 'blue':
        color = (255, 100, 50)  # Azul en BGR
    elif meeple_color == 'black':
        color = (50, 50, 50)    # Negro/Gris oscuro
    else:
        color = (100, 100, 255) # Rojo para unknown

    # Dibujar meeple (círculo)
    cv2.circle(base_image, (center_x, center_y), radius, color, -1)

    # Agregar borde blanco para simular el borde del meeple
    cv2.circle(base_image, (center_x, center_y), radius, (255, 255, 255), 2)

    # Guardar imagen
    success = cv2.imwrite(output_path, base_image)
    if success:
        print(f"✅ Imagen creada: {output_path}")
        print(f"   Meeple {meeple_color} en posición {position} (centro: {center_x}, {center_y})")
    else:
        print(f"❌ Error al guardar: {output_path}")

    return success

def create_multi_meeple_image(base_image_path: str, output_path: str):
    """
    Crea una imagen de prueba con múltiples meeples
    """
    # Cargar imagen base
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        print(f"❌ No se pudo cargar: {base_image_path}")
        return False

    height, width = base_image.shape[:2]

    # Definir múltiples meeples: (color, position)
    meeples = [
        ('blue', 0),    # Azul en esquina superior izquierda
        ('black', 2),   # Negro en esquina superior derecha
        ('blue', 6),    # Azul en esquina inferior izquierda
    ]

    for meeple_color, position in meeples:
        # Calcular posición del meeple basada en la cuadrícula 3x3
        row = position // 3
        col = position % 3

        # Centro de cada región
        center_x = int((col + 0.5) * width / 3)
        center_y = int((row + 0.5) * height / 3)

        # Radio del meeple
        radius = min(width, height) // 12

        # Color del meeple
        if meeple_color == 'blue':
            color = (255, 100, 50)  # Azul en BGR
        elif meeple_color == 'black':
            color = (50, 50, 50)    # Negro/Gris oscuro
        else:
            color = (100, 100, 255) # Rojo para unknown

        # Dibujar meeple (círculo)
        cv2.circle(base_image, (center_x, center_y), radius, color, -1)

        # Agregar borde blanco para simular el borde del meeple
        cv2.circle(base_image, (center_x, center_y), radius, (255, 255, 255), 2)

    # Guardar imagen
    success = cv2.imwrite(output_path, base_image)
    if success:
        print(f"✅ Imagen con múltiples meeples creada: {output_path}")
        print(f"   Meeples: {meeples}")
    else:
        print(f"❌ Error al guardar: {output_path}")

    return success

def create_multiple_test_images():
    """Crea múltiples imágenes de prueba"""

    # Buscar imágenes base
    tiles_dir = Path("tiles")
    base_images = list(tiles_dir.glob("*.png"))

    if not base_images:
        print("❌ No se encontraron imágenes base en 'tiles/'")
        return

    print(f"📸 Creando imágenes de prueba usando {len(base_images)} losetas base")

    # Crear directorio para imágenes de prueba
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)

    created = 0

    # Para cada imagen base, crear versiones con meeples
    for base_img in base_images[:2]:  # Solo usar las primeras 2 para no crear demasiadas
        base_name = base_img.stem

        # Crear imágenes con diferentes meeples
        test_cases = [
            ('blue', 0),   # Azul en esquina superior izquierda
            ('blue', 4),   # Azul en centro
            ('black', 2),  # Negro en esquina superior derecha
            ('black', 8),  # Negro en esquina inferior derecha
        ]

        for color, position in test_cases:
            output_name = f"{base_name}_{color}_pos{position}.png"
            output_path = test_dir / output_name

            if create_test_image_with_meeple(str(base_img), str(output_path), color, position):
                created += 1

    # Crear imagen con múltiples meeples
    if base_images:
        base_img = str(base_images[0])
        output_path = str(test_dir / "B_multi_meeples.png")

        if create_multi_meeple_image(base_img, output_path):
            created += 1

    print(f"\n✅ Creadas {created} imágenes de prueba en '{test_dir}/'")
    print("\nPuedes probar el detector con:")
    print("python test_detector.py test_images/imagen_prueba.png")

def main():
    print("🎨 CREADOR DE IMÁGENES DE PRUEBA")
    print("=" * 40)

    create_multiple_test_images()

    print("\n💡 Estas imágenes te permiten probar:")
    print("- Detección de círculos")
    print("- Clasificación de colores")
    print("- Determinación de posiciones")
    print("- Visualización de resultados")

if __name__ == "__main__":
    main()