#!/usr/bin/env python3
"""
Script para probar el detector con imágenes reales del usuario
"""

from pathlib import Path
from src.meeple_detector_cv import MeepleDetector
import json

def test_real_images():
    """Prueba el detector con imágenes reales"""

    detector = MeepleDetector()

    # Directorio de imágenes reales
    real_images_dir = Path("real_test_images")
    if not real_images_dir.exists():
        print("❌ Directorio 'real_test_images' no encontrado.")
        print("   Por favor, coloca tus imágenes de prueba en la carpeta 'real_test_images'")
        return

    # Obtener todas las imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(real_images_dir.glob(ext))

    if not image_files:
        print("❌ No se encontraron imágenes en 'real_test_images'")
        print("   Coloca tus fotos de losetas de Carcassonne con meeples en esta carpeta")
        return

    print(f"🔍 Probando detector con {len(image_files)} imágenes reales...")

    results = []
    total_meeples = 0

    for image_file in sorted(image_files):
        print(f"\n📸 Probando: {image_file.name}")

        try:
            result = detector.process_image(image_file)

            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue

            results.append(result)

            meeples_found = result['meeples_found']
            print(f"✅ Detectados: {meeples_found} meeples")

            if meeples_found > 0:
                total_meeples += meeples_found
                for i, meeple in enumerate(result['meeples'], 1):
                    print(f"   {i}. Meeple {meeple['color']} en posición {meeple['position']}")

                # Generar visualización
                vis_path = f"visualizations/{image_file.stem}_real_detection.png"
                detector.visualize_detection(str(image_file), vis_path)
                print(f"   📊 Visualización guardada: {vis_path}")

        except Exception as e:
            print(f"❌ Error procesando {image_file.name}: {e}")

    print(f"\n{'='*50}")
    print("📊 RESULTADOS CON IMÁGENES REALES")
    print(f"{'='*50}")
    print(f"Imágenes procesadas: {len(results)}")
    print(f"Total meeples detectados: {total_meeples}")

    if results:
        # Guardar resultados
        output_file = 'real_test_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\n💾 Resultados guardados en: {output_file}")

    print(f"\n💡 Si los resultados no son precisos, podemos ajustar los parámetros:")
    print(f"   - Ejecuta: python tune_params.py")
    print(f"   - O modifica los parámetros en src/meeple_detector_cv.py")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)

if __name__ == "__main__":
    test_real_images()