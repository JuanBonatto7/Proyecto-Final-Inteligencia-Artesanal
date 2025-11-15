#!/usr/bin/env python3
"""
Script para agregar rotaciones de 90°, 180° y 270° a cada imagen en las carpetas de referencias
"""
import cv2
import numpy as np
from pathlib import Path
import os

def add_rotations_to_references():
    """Agregar rotaciones a todas las imágenes de referencia"""

    reference_folder = Path("referencias_organizadas")

    if not reference_folder.exists():
        print(f"Error: Carpeta {reference_folder} no existe")
        return

    total_added = 0

    # Para cada letra A-Z
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        letter_folder = reference_folder / letter

        if not letter_folder.exists():
            print(f"Carpeta {letter} no existe, saltando...")
            continue

        print(f"Procesando carpeta {letter}...")

        # Obtener todas las imágenes PNG y JPG
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(letter_folder.glob(ext))

        for img_path in image_files:
            # Saltar si ya es una rotación (tiene _rot en el nombre)
            if '_rot' in img_path.stem:
                continue

            print(f"  Procesando {img_path.name}")

            # Leer imagen
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"    Error cargando {img_path}")
                continue

            # Nombre base sin extensión
            base_name = img_path.stem
            extension = img_path.suffix

            # Rotar 90°
            rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rot90_path = letter_folder / f"{base_name}_rot90{extension}"
            cv2.imwrite(str(rot90_path), rotated_90)
            print(f"    Guardado {rot90_path.name}")
            total_added += 1

            # Rotar 180°
            rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
            rot180_path = letter_folder / f"{base_name}_rot180{extension}"
            cv2.imwrite(str(rot180_path), rotated_180)
            print(f"    Guardado {rot180_path.name}")
            total_added += 1

            # Rotar 270°
            rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rot270_path = letter_folder / f"{base_name}_rot270{extension}"
            cv2.imwrite(str(rot270_path), rotated_270)
            print(f"    Guardado {rot270_path.name}")
            total_added += 1

    print(f"\nTotal de imágenes rotadas agregadas: {total_added}")

if __name__ == "__main__":
    add_rotations_to_references()