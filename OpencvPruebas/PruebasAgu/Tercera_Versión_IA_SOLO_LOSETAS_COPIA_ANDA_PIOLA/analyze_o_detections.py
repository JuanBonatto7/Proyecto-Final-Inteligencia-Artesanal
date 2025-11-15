#!/usr/bin/env python3
"""
Script para inspeccionar imágenes que dieron detecciones de O
"""
import cv2
import numpy as np
from pathlib import Path

def analyze_image(image_path):
    """Analiza una imagen y muestra sus características"""
    print(f"\nAnalizando: {image_path.name}")

    # Cargar imagen
    image = cv2.imread(str(image_path))
    if image is None:
        print("No se pudo cargar la imagen")
        return

    # Redimensionar como en el detector
    image_resized = cv2.resize(image, (200, 200))

    # Convertir a escala de grises
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Calcular estadísticas
    total_pixels = gray.size
    white_pixels_150 = np.sum(gray > 150)
    white_pixels_180 = np.sum(gray > 180)
    white_pixels_200 = np.sum(gray > 200)
    white_ratio_150 = white_pixels_150 / total_pixels
    white_ratio_180 = white_pixels_180 / total_pixels
    white_ratio_200 = white_pixels_200 / total_pixels
    variance = np.var(gray)
    mean_intensity = np.mean(gray)

    print(f"  Dimensiones originales: {image.shape}")
    print(f"  Intensidad media: {mean_intensity:.1f}")
    print(f"  Varianza: {variance:.1f}")
    print(f"  Ratio blanco (>150): {white_ratio_150:.3f}")
    print(f"  Ratio blanco (>180): {white_ratio_180:.3f}")
    print(f"  Ratio blanco (>200): {white_ratio_200:.3f}")
    print(f"  ¿Es blanco (threshold 150)? {white_ratio_150 > 0.6 and variance < 3000}")
    print(f"  ¿Es blanco (threshold 180)? {white_ratio_180 > 0.6 and variance < 3000}")

def main():
    # Algunas imágenes que dieron O en el test anterior
    test_images = [
        "tile_057_r7_c2.png",
        "tile_073_r9_c2_1.png",
        "tile_020_r-1_c11.png",
        "tile_021_r3_c3_2.png",
        "tile_017_r2_c1.png"
    ]

    base_path = Path(r"C:\Users\agust\OneDrive\Desktop\Uni\Proyecto\Proyecto-Final-Inteligencia-Artesanal\OpencvPruebas\PruebasAgu\Segunda_version_IA\dataset\unlabeled")

    for img_name in test_images:
        img_path = base_path / img_name
        if img_path.exists():
            analyze_image(img_path)
        else:
            print(f"Imagen no encontrada: {img_name}")

if __name__ == "__main__":
    main()