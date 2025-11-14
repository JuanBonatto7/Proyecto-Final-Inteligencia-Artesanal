#!/usr/bin/env python3
"""
Sistema Integrado de Detección de Meeples
Combina anotación manual, evaluación y mejora del detector
"""

import os
import json
from pathlib import Path
import subprocess
import sys

def print_header():
    """Imprimir header del sistema"""
    print("🎯 Sistema Integrado de Detección de Meeples")
    print("=" * 50)
    print("Herramientas disponibles:")
    print("1. 📝 Anotación Manual (meeple_annotator.py)")
    print("2. 🔍 Evaluación del Detector (evaluate_detector.py)")
    print("3. 🧠 Detector CNN (cnn_meeple_detector.py)")
    print("4. ⚙️ Configuración del Detector OpenCV")
    print("5. 📊 Ver Resultados")
    print("6. ❌ Salir")
    print()

def check_dependencies():
    """Verificar dependencias necesarias"""
    required_modules = ['cv2', 'numpy', 'torch', 'matplotlib', 'sklearn']
    missing = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print("⚠️ Módulos faltantes:"        for mod in missing:
            print(f"   - {mod}")
        print("Instala con: pip install " + " ".join(missing))
        return False

    return True

def run_annotator():
    """Ejecutar herramienta de anotación"""
    print("📝 Iniciando anotación manual...")
    print("Instrucciones:")
    print("- Click para marcar meeples (azul/negro)")
    print("- ESPACIO para siguiente imagen")
    print("- Q para salir")
    print()

    try:
        subprocess.run([sys.executable, 'meeple_annotator.py'])
    except KeyboardInterrupt:
        print("\n⏹️ Anotación interrumpida")

def run_evaluator():
    """Ejecutar evaluación del detector"""
    annotations_file = 'manual_annotations.json'

    if not Path(annotations_file).exists():
        print(f"❌ No hay anotaciones. Ejecuta primero la opción 1.")
        return

    print("🔍 Evaluando detector OpenCV vs ground truth...")
    try:
        subprocess.run([sys.executable, 'evaluate_detector.py'])
    except KeyboardInterrupt:
        print("\n⏹️ Evaluación interrumpida")

def run_cnn_detector():
    """Ejecutar detector CNN"""
    print("🧠 Detector CNN...")
    try:
        subprocess.run([sys.executable, 'cnn_meeple_detector.py'])
    except KeyboardInterrupt:
        print("\n⏹️ CNN interrumpido")

def configure_detector():
    """Configurar parámetros del detector OpenCV"""
    print("⚙️ Configuración del Detector OpenCV")
    print("=" * 40)

    # Mostrar configuración actual
    try:
        with open('src/meeple_detector_cv.py', 'r') as f:
            content = f.read()

        # Extraer rangos de color
        if 'color_ranges' in content:
            print("Configuración actual:")
            print("- Rangos HSV ajustados para valores exactos proporcionados")
            print("- Azul: HSV(212, 64%, 62%)")
            print("- Negro: HSV(240, 10%, 8%)")
            print("- Rango azul: H[95-115], S[140-180], V[120-190]")
            print("- Rango negro: H[0-179], S[0-50], V[0-50]")
        else:
            print("Configuración no encontrada")

    except Exception as e:
        print(f"Error leyendo configuración: {e}")

    print("\n💡 Para modificar parámetros, edita src/meeple_detector_cv.py")

def show_results():
    """Mostrar resultados disponibles"""
    print("📊 Resultados Disponibles")
    print("=" * 30)

    result_files = [
        'manual_annotations.json',
        'evaluation_results.json',
        'real_test_results.json',
        'detection_results.json',
        'best_meeple_cnn.pth'
    ]

    for file in result_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} (no existe)")

    # Mostrar resumen si existe
    eval_file = 'evaluation_results.json'
    if Path(eval_file).exists():
        try:
            with open(eval_file, 'r') as f:
                results = json.load(f)

            print("
📈 Resumen de Evaluación:"            print(f"   Imágenes: {results.get('total_images', 0)}")
            print(f"   Precisión: {results.get('overall_precision', 0):.3f}")
            print(f"   Recall: {results.get('overall_recall', 0):.3f}")
            print(f"   F1-Score: {results.get('f1_score', 0):.3f}")
        except:
            print("   Error leyendo resultados")

def main():
    """Función principal"""
    if not check_dependencies():
        return

    while True:
        print_header()
        choice = input("Elige una opción (1-6): ").strip()

        if choice == '1':
            run_annotator()

        elif choice == '2':
            run_evaluator()

        elif choice == '3':
            run_cnn_detector()

        elif choice == '4':
            configure_detector()

        elif choice == '5':
            show_results()

        elif choice == '6':
            print("👋 ¡Hasta luego!")
            break

        else:
            print("❌ Opción inválida")

        input("\nPresiona ENTER para continuar...")

if __name__ == "__main__":
    main()