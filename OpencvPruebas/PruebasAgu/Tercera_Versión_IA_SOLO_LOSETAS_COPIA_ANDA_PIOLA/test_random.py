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
    unlabeled_path = Path(r"C:\Users\agust\OneDrive\Desktop\PruebasAgu\Segunda_version_IA\dataset\unlabeled")

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
                sys.executable, "pipeline.py", str(img_path)
            ], capture_output=True, text=True, timeout=120, cwd=detector_dir)

            # Debug: imprimir salida completa si hay error
            if result.returncode != 0:
                print(f"Error en subprocess: {result.stderr}")
                results.append((img_path.name, "ERROR", None))
                continue

            # Extraer el resultado
            output_lines = result.stdout.strip().split('\n')
            tile_type = None
            confidence = None
            rotation = None
            for line in output_lines:
                if "Tipo detectado:" in line:
                    # Parse "Tipo detectado: M"
                    parts = line.split()
                    tile_type = parts[2]
                    confidence = None  # No hay confianza en pipeline
                elif "Rotación detectada:" in line:
                    # Parse "Rotación detectada: 270°"
                    parts = line.split()
                    rot_str = parts[2].strip('°')
                    try:
                        rotation = int(rot_str)
                    except ValueError:
                        rotation = None
            
            if tile_type:
                results.append((img_path.name, tile_type, confidence, rotation))
                conf_str = f" (confianza: {confidence:.2f})" if confidence else ""
                rot_str = f", rotación {rotation}°" if rotation is not None else ""
                print(f"Resultado: {tile_type}{conf_str}{rot_str}")
            else:
                print(f"No se encontró resultado en la salida. Salida completa:\n{result.stdout}")
                results.append((img_path.name, "NO_RESULT", None, None))

        except subprocess.TimeoutExpired:
            print("Timeout en la detección")
            results.append((img_path.name, "TIMEOUT", None, None))
        except Exception as e:
            print(f"Error: {e}")
            results.append((img_path.name, "ERROR", None, None))

    # Resumen final
    print(f"\n{'='*50}")
    print("RESUMEN DE PRUEBAS:")
    print(f"{'='*50}")

    # Contar tipos detectados
    from collections import Counter
    tile_counts = Counter(result[1] for result in results if result[1] not in ["ERROR", "TIMEOUT", "NO_RESULT"])
    
    # Recopilar confianzas (solo para detecciones no blancas)
    confidences = [result[2] for result in results if result[2] is not None and result[1] not in ["ERROR", "TIMEOUT", "NO_RESULT", "BLANCO"]]

    # Recopilar rotaciones
    rotations = [result[3] for result in results if result[3] is not None and result[1] not in ["ERROR", "TIMEOUT", "NO_RESULT", "BLANCO"]]

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

    if rotations:
        print(f"\nEstadísticas de rotación (excluyendo blancas):")
        rotation_counts = Counter(rotations)
        for rot, count in sorted(rotation_counts.items()):
            print(f"  {rot}°: {count}")
        print(f"  Total con rotación detectada: {len(rotations)}")

    print("\nDistribución de losetas detectadas:")
    for tile, count in sorted(tile_counts.items()):
        print(f"  {tile}: {count}")

    # Mostrar algunos resultados con confianza
    print("\nPrimeros 10 resultados:")
    for img, tile, conf, rot in results[:10]:
        if tile == "BLANCO":
            conf_str = ""
            rot_str = ""
        else:
            conf_str = f" (confianza: {conf:.2f})" if conf is not None else ""
            rot_str = f", rotación {rot}°" if rot is not None else ""
        print(f"  {img} -> {tile}{conf_str}{rot_str}")

if __name__ == "__main__":
    test_random_images(50)