#!/usr/bin/env python3
"""
Script para probar el detector con 50 imágenes aleatorias del dataset unlabeled
"""
import os
import random
import subprocess
import sys
from pathlib import Path

def test_random_images(num_images=50):
    """Prueba el detector con imágenes aleatorias del dataset unlabeled"""

    # Ruta al dataset unlabeled
    unlabeled_path = Path(r"C:\Users\agust\OneDrive\Desktop\Uni\Proyecto\Proyecto-Final-Inteligencia-Artesanal\OpencvPruebas\PruebasAgu\Segunda_version_IA\dataset\unlabeled")

    if not unlabeled_path.exists():
        print(f"Error: No se encuentra el directorio {unlabeled_path}")
        return

    # Obtener todas las imágenes PNG
    image_files = list(unlabeled_path.glob("*.png"))

    if len(image_files) < num_images:
        print(f"Advertencia: Solo hay {len(image_files)} imágenes, probando todas")
        num_images = len(image_files)

    # Seleccionar imágenes aleatorias
    random_images = random.sample(image_files, num_images)

    print(f"Probando {num_images} imágenes aleatorias...")

    # Directorio del detector
    detector_dir = Path(__file__).parent

    results = []
    for i, img_path in enumerate(random_images, 1):
        print(f"\n--- Prueba {i}/{num_images}: {img_path.name} ---")

        # Ejecutar el detector
        try:
            result = subprocess.run([
                sys.executable, "tile_detector.py", str(img_path)
            ], capture_output=True, text=True, timeout=60, cwd=detector_dir)

            # Debug: imprimir salida completa si hay error
            if result.returncode != 0:
                print(f"Error en subprocess: {result.stderr}")
                results.append((img_path.name, "ERROR", None))
                continue

            # Extraer el resultado
            output_lines = result.stdout.strip().split('\n')
            tile_type = None
            confidence = None
            for line in output_lines:
                if "Loseta detectada:" in line:
                    # Parse "Loseta detectada: K (confianza: 0.90)" o "Loseta detectada: BLANCO (sin loseta)"
                    parts = line.split()
                    tile_type = parts[2]  # K o BLANCO
                    if tile_type == "BLANCO":
                        confidence = None  # No hay confianza para blancas
                    elif len(parts) > 4:
                        confidence_str = parts[4].strip('()')  # 0.90
                        try:
                            confidence = float(confidence_str)
                        except ValueError:
                            confidence = None
                    break
            
            if tile_type:
                results.append((img_path.name, tile_type, confidence))
                print(f"Resultado: {tile_type}" + (f" (confianza: {confidence:.2f})" if confidence else ""))
            else:
                print(f"No se encontró 'Loseta detectada:' en la salida. Salida completa:\n{result.stdout}")
                results.append((img_path.name, "NO_RESULT", None))

        except subprocess.TimeoutExpired:
            print("Timeout en la detección")
            results.append((img_path.name, "TIMEOUT", None))
        except Exception as e:
            print(f"Error: {e}")
            results.append((img_path.name, "ERROR", None))

    # Resumen final
    print(f"\n{'='*50}")
    print("RESUMEN DE PRUEBAS:")
    print(f"{'='*50}")

    # Contar tipos detectados
    from collections import Counter
    tile_counts = Counter(result[1] for result in results if result[1] not in ["ERROR", "TIMEOUT", "NO_RESULT"])
    
    # Recopilar confianzas (solo para detecciones no blancas)
    confidences = [result[2] for result in results if result[2] is not None and result[1] not in ["ERROR", "TIMEOUT", "NO_RESULT", "BLANCO"]]

    print(f"Total imágenes probadas: {len(results)}")
    print(f"Éxitos: {len([r for r in results if r[1] not in ['ERROR', 'TIMEOUT', 'NO_RESULT']])}")
    print(f"Errores: {len([r for r in results if r[1] == 'ERROR'])}")
    print(f"Timeouts: {len([r for r in results if r[1] == 'TIMEOUT'])}")
    print(f"Sin resultado: {len([r for r in results if r[1] == 'NO_RESULT'])}")
    print(f"Áreas blancas: {len([r for r in results if r[1] == 'BLANCO'])}")

    if confidences:
        print(f"\nEstadísticas de confianza (excluyendo blancas):")
        print(f"  Media: {sum(confidences)/len(confidences):.3f}")
        print(f"  Máxima: {max(confidences):.3f}")
        print(f"  Mínima: {min(confidences):.3f}")
        print(f"  > 0.8: {len([c for c in confidences if c > 0.8])}")
        print(f"  > 0.9: {len([c for c in confidences if c > 0.9])}")

    print("\nDistribución de losetas detectadas:")
    for tile, count in sorted(tile_counts.items()):
        print(f"  {tile}: {count}")

    # Mostrar algunos resultados con confianza
    print("\nPrimeros 10 resultados:")
    for img, tile, conf in results[:10]:
        if tile == "BLANCO":
            conf_str = ""
        else:
            conf_str = f" (confianza: {conf:.2f})" if conf is not None else ""
        print(f"  {img} -> {tile}{conf_str}")

if __name__ == "__main__":
    test_random_images(50)