#!/usr/bin/env python3
"""
Script para probar todas las referencias y verificar que no hay confusiones
"""
import os
import subprocess
import sys

def test_all_references():
    """Prueba todas las referencias y verifica detección correcta"""
    referencias_dir = "referencias"
    results = {}

    if not os.path.exists(referencias_dir):
        print(f"Directorio {referencias_dir} no encontrado")
        return

    # Obtener lista de archivos de referencia
    ref_files = [f for f in os.listdir(referencias_dir) if f.endswith('.png') and len(f) <= 6]  # A.png to X.png
    ref_files.sort()

    print(f"Probando {len(ref_files)} referencias...")

    for ref_file in ref_files:
        expected = ref_file[0]  # A, B, C, etc.
        full_path = os.path.join(referencias_dir, ref_file)

        try:
            # Ejecutar el detector
            result = subprocess.run([sys.executable, 'tile_detector.py', full_path],
                                  capture_output=True, text=True, timeout=30)

            # Extraer la detección del output
            output_lines = result.stdout.strip().split('\n')
            detected = None
            for line in output_lines:
                if "Loseta detectada:" in line:
                    # Extraer la letra de "Loseta detectada: X (confianza: 0.99)"
                    detected = line.split("Loseta detectada:")[1].strip().split()[0]
                    break

            if detected:
                results[expected] = detected
                status = "✓" if detected == expected else "✗"
                print(f"{status} {expected} -> {detected}")
            else:
                print(f"✗ {expected} -> ERROR: No se pudo detectar")
                results[expected] = "ERROR"

        except subprocess.TimeoutExpired:
            print(f"✗ {expected} -> TIMEOUT")
            results[expected] = "TIMEOUT"
        except Exception as e:
            print(f"✗ {expected} -> ERROR: {str(e)}")
            results[expected] = "ERROR"

    # Resumen
    print("\n" + "="*50)
    print("RESUMEN DE PRUEBAS:")
    print("="*50)

    correct = 0
    total = len(results)

    for expected, detected in results.items():
        if detected == expected:
            correct += 1
        elif detected in ["ERROR", "TIMEOUT"]:
            print(f"❌ {expected}: {detected}")
        else:
            print(f"❌ {expected}: Detectado como {detected} (esperado {expected})")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(".1f")

    if accuracy >= 95:
        print("🎉 ¡Excelente precisión!")
    elif accuracy >= 90:
        print("✅ Buena precisión")
    else:
        print("⚠️  Precisión baja - revisar algoritmo")

    return results

if __name__ == "__main__":
    test_all_references()