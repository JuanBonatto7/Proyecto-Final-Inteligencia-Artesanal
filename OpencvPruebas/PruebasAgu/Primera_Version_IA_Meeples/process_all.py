#!/usr/bin/env python3
"""
Script para procesar todas las imágenes y generar estadísticas
"""

from pathlib import Path
from src.meeple_detector_cv import MeepleDetector
import json
from collections import defaultdict
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def main():
    detector = MeepleDetector()

    # Directorio de imágenes
    tiles_dir = Path("tiles")
    if not tiles_dir.exists():
        print("❌ Directorio 'tiles' no encontrado. Asegúrate de tener las imágenes en la carpeta 'tiles'")
        return

    # Obtener todas las imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(tiles_dir.glob(ext))

    if not image_files:
        print("❌ No se encontraron imágenes en la carpeta 'tiles'")
        return

    print(f"🔍 Procesando {len(image_files)} imágenes...")

    # Procesar todas las imágenes
    results = []
    stats = {
        'total_images': len(image_files),
        'images_with_meeples': 0,
        'total_meeples': 0,
        'meeples_by_color': defaultdict(int),
        'meeples_by_position': defaultdict(int),
        'detection_errors': 0
    }

    for image_file in sorted(image_files):
        print(f"  Procesando: {image_file.name}", end=' ... ')

        try:
            result = detector.process_image(image_file)

            if 'error' in result:
                print("❌ Error")
                stats['detection_errors'] += 1
                continue

            results.append(result)

            meeples_found = result['meeples_found']
            print(f"✅ {meeples_found} meeples")

            if meeples_found > 0:
                stats['images_with_meeples'] += 1
                stats['total_meeples'] += meeples_found

                for meeple in result['meeples']:
                    stats['meeples_by_color'][meeple['color']] += 1
                    if meeple['position'] != -1:
                        stats['meeples_by_position'][meeple['position']] += 1

        except Exception as e:
            print(f"❌ Error: {e}")
            stats['detection_errors'] += 1

    # Mostrar estadísticas
    print(f"\n{'='*50}")
    print("📊 ESTADÍSTICAS FINALES")
    print(f"{'='*50}")
    print(f"Total de imágenes procesadas: {stats['total_images']}")
    print(f"Imágenes con meeples: {stats['images_with_meeples']}")
    print(f"Total de meeples detectados: {stats['total_meeples']}")
    print(f"Errores de detección: {stats['detection_errors']}")

    print(f"\n📈 DISTRIBUCIÓN POR COLOR:")
    for color, count in stats['meeples_by_color'].items():
        print(f"  {color.capitalize()}: {count}")

    print(f"\n📍 DISTRIBUCIÓN POR POSICIÓN:")
    print("┌───┬───┬───┐")
    for i in range(3):
        row = "│"
        for j in range(3):
            pos = i * 3 + j
            count = stats['meeples_by_position'][pos]
            row += f" {count:2d}│"
        print(row)
        if i < 2:
            print("├───┼───┼───┤")
    print("└───┴───┴───┘")

    # Guardar resultados detallados
    output_file = 'detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n💾 Resultados detallados guardados en: {output_file}")

    # Crear visualizaciones para algunas imágenes
    print("\n🎨 Generando visualizaciones de ejemplo...")
    vis_dir = Path("visualizations")
    vis_dir.mkdir(exist_ok=True)

    # Visualizar las primeras 5 imágenes con meeples
    images_with_meeples = [r for r in results if r['meeples_found'] > 0][:5]

    for result in images_with_meeples:
        image_path = result['image_path']
        image_name = Path(image_path).stem
        vis_path = vis_dir / f"{image_name}_detection.png"

        print(f"  Generando: {vis_path.name}")
        detector.visualize_detection(image_path, str(vis_path))

    print(f"\n✅ Visualizaciones guardadas en: {vis_dir}/")

if __name__ == "__main__":
    main()