#!/usr/bin/env python3
"""
Script para analizar colores en imágenes reales y ajustar rangos de detección
"""

from pathlib import Path
from src.meeple_detector_cv import MeepleDetector
import cv2
import numpy as np
import json

def analyze_circle_colors(image_path: str):
    """Analiza los colores de los círculos detectados en una imagen"""

    detector = MeepleDetector()
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"❌ No se pudo cargar: {image_path}")
        return None

    # Detectar círculos
    circles = detector.detect_circles(image)

    if len(circles) == 0:
        print(f"⚠️  No se detectaron círculos en: {Path(image_path).name}")
        return None

    print(f"🔍 Analizando {len(circles)} círculos en: {Path(image_path).name}")

    # Convertir a HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    circle_colors = []

    for x, y, r in circles:
        # Extraer región del círculo (un poco más pequeña para evitar bordes)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), int(r * 0.8), 255, -1)

        # Calcular colores promedio en la región
        mean_hsv = cv2.mean(hsv_image, mask=mask)
        mean_bgr = cv2.mean(image, mask=mask)

        circle_colors.append({
            'center': (x, y),
            'radius': r,
            'hsv': mean_hsv[:3],  # H, S, V
            'bgr': mean_bgr[:3],  # B, G, R
            'brightness': mean_hsv[2],  # V channel
            'saturation': mean_hsv[1]   # S channel
        })

        print(f"  Círculo en ({x}, {y}): HSV={mean_hsv[:3]}, BGR={mean_bgr[:3]}")

    return circle_colors

def analyze_multiple_images():
    """Analiza colores en múltiples imágenes reales"""

    real_dir = Path("real_test_images")
    if not real_dir.exists():
        print("❌ Directorio 'real_test_images' no encontrado")
        return

    # Tomar solo las primeras 10 imágenes para análisis
    image_files = list(real_dir.glob("*.jpg"))[:10]

    if not image_files:
        print("❌ No se encontraron imágenes JPG")
        return

    print(f"🎨 ANALIZANDO COLORES EN {len(image_files)} IMÁGENES REALES")
    print("=" * 60)

    all_circle_colors = []

    for image_file in image_files:
        colors = analyze_circle_colors(image_file)
        if colors:
            all_circle_colors.extend(colors)

    if not all_circle_colors:
        print("❌ No se detectaron círculos en ninguna imagen")
        return

    print(f"\n📊 ANÁLISIS DE {len(all_circle_colors)} CÍRCULOS DETECTADOS")
    print("=" * 60)

    # Extraer valores HSV
    h_values = [c['hsv'][0] for c in all_circle_colors]
    s_values = [c['hsv'][1] for c in all_circle_colors]
    v_values = [c['hsv'][2] for c in all_circle_colors]

    print("📈 ESTADÍSTICAS HSV:")
    print(f"  Hue (H):     min={min(h_values):.1f}, max={max(h_values):.1f}, avg={np.mean(h_values):.1f}")
    print(f"  Saturation:  min={min(s_values):.1f}, max={max(s_values):.1f}, avg={np.mean(s_values):.1f}")
    print(f"  Value (V):   min={min(v_values):.1f}, max={max(v_values):.1f}, avg={np.mean(v_values):.1f}")

    # Sugerir rangos basados en los datos
    print("\n💡 RANGOS SUGERIDOS PARA AJUSTE:")
    print(f"  Azul aproximado: H=[{max(0, np.mean(h_values)-30):.0f}, {min(180, np.mean(h_values)+30):.0f}], S>30, V>40")
    print(f"  Negro aproximado: V<50, S<50")

    # Guardar datos para análisis posterior
    with open('color_analysis.json', 'w') as f:
        json.dump({
            'total_circles': len(all_circle_colors),
            'h_values': h_values,
            's_values': s_values,
            'v_values': v_values,
            'suggested_ranges': {
                'blue_h_range': [max(0, np.mean(h_values)-30), min(180, np.mean(h_values)+30)],
                'black_v_threshold': 50,
                'black_s_threshold': 50
            }
        }, f, indent=2)

    print("\n💾 Datos guardados en: color_analysis.json")
    print("\n🔧 Para ajustar, modifica src/meeple_detector_cv.py en la sección color_ranges")

def main():
    analyze_multiple_images()

if __name__ == "__main__":
    main()