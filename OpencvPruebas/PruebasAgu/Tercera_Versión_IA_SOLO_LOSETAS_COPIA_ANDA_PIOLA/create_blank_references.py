#!/usr/bin/env python3
"""
Script para generar imágenes blancas de referencia para la clase BLANCO
"""
import cv2
import numpy as np
import os
from pathlib import Path

def create_blank_references(output_dir, num_images=20):
    """Genera imágenes blancas de referencia con variaciones"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Generando {num_images} imágenes blancas en {output_dir}")

    for i in range(num_images):
        # Crear imagen blanca base (200x200 para que coincida con el procesamiento)
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # Blanco puro

        # Agregar variaciones menores para que el modelo aprenda
        if i % 4 == 0:
            # Variación 1: Blanco con ruido mínimo
            noise = np.random.normal(0, 2, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        elif i % 4 == 1:
            # Variación 2: Blanco ligeramente grisáceo
            img = np.ones((200, 200, 3), dtype=np.uint8) * 240
        elif i % 4 == 2:
            # Variación 3: Blanco con gradiente sutil
            for y in range(200):
                factor = 1.0 - (y / 200) * 0.1  # Gradiente del 100% al 90%
                img[y, :] = img[y, :] * factor
        else:
            # Variación 4: Blanco puro
            pass

        # Asegurar que se mantenga en rango válido
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Guardar imagen
        filename = f"BLANCO_ref_{i:03d}.png"
        filepath = output_path / filename
        print(f"  Guardando en: {filepath}")
        success = cv2.imwrite(str(filepath), img)
        if success:
            print(f"  Generada: {filename}")
        else:
            print(f"  ERROR: No se pudo guardar {filename}")

    print(f"Generación completada: {num_images} imágenes blancas creadas")

if __name__ == "__main__":
    output_dir = r"referencias_organizadas\BLANCO"
    create_blank_references(output_dir, 20)