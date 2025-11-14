"""
Ejemplo de Uso Completo del Sistema de IA

Este script demuestra cómo usar el sistema completo para clasificar losetas.
"""

import os
import json
from pathlib import Path


def ejemplo_completo():
    """Ejemplo de flujo completo desde detección hasta clasificación."""
    
    print("\n" + "="*70)
    print("EJEMPLO COMPLETO: DETECCIÓN + CLASIFICACIÓN DE LOSETAS")
    print("="*70 + "\n")
    
    # ===== PARTE 1: DETECCIÓN DE LOSETAS =====
    print("📸 PASO 1: Detección de losetas en el tablero")
    print("-" * 70)
    
    # Ruta a la imagen del tablero
    board_image = "foto_tablero.jpg"
    
    if not os.path.exists(board_image):
        print(f"⚠️ No se encontró {board_image}")
        print("Coloca una foto de un tablero de Carcassonne en el directorio actual")
        return
    
    print(f"Imagen del tablero: {board_image}")
    print("Ejecutando detector...")
    
    # Aquí normalmente ejecutarías el detector
    # os.system(f'python carcassonne.py {board_image}')
    
    # Para este ejemplo, asumimos que ya tenemos las losetas en tiles/
    tiles_dir = "tiles"
    
    if not os.path.exists(tiles_dir):
        print(f"⚠️ Directorio {tiles_dir}/ no encontrado")
        print(f"Primero ejecuta el detector: python carcassonne.py {board_image}")
        return
    
    print(f"✓ Losetas extraídas en {tiles_dir}/\n")
    
    # ===== PARTE 2: CLASIFICACIÓN CON IA =====
    print("🤖 PASO 2: Clasificación con IA")
    print("-" * 70)
    
    model_path = "models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo no encontrado: {model_path}")
        print("Primero entrena un modelo con: python train.py ...")
        return
    
    print(f"Modelo: {model_path}")
    print("Clasificando losetas...\n")
    
    # Importar clasificador
    from inference import TileClassifier
    
    # Crear clasificador
    classifier = TileClassifier(model_path)
    
    # Clasificar todas las losetas
    results = classifier.predict_directory(
        directory=tiles_dir,
        output_file='board_classification.json',
        batch_size=32
    )
    
    print(f"✓ {len(results)} losetas clasificadas\n")
    
    # ===== PARTE 3: ANÁLISIS DE RESULTADOS =====
    print("📊 PASO 3: Análisis de resultados")
    print("-" * 70)
    
    # Contar tipos de losetas
    tile_counts = {}
    for result in results:
        letter = result['tile_letter']
        tile_counts[letter] = tile_counts.get(letter, 0) + 1
    
    print("\n🎴 Distribución de tipos de losetas:")
    for letter in sorted(tile_counts.keys()):
        count = tile_counts[letter]
        bar = "█" * count
        print(f"  {letter:7s} [{count:2d}]: {bar}")
    
    # Contar rotaciones
    rotation_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for result in results:
        rotation_counts[result['rotation']] += 1
    
    print("\n🔄 Distribución de rotaciones:")
    for rotation, count in rotation_counts.items():
        degrees = rotation * 90
        bar = "█" * (count // 2)
        print(f"  {degrees:3d}° [{count:2d}]: {bar}")
    
    # Contar meeples
    meeple_count = sum(1 for r in results if r['has_meeple'])
    no_meeple_count = len(results) - meeple_count
    
    print(f"\n👥 Losetas con meeple: {meeple_count}/{len(results)}")
    print(f"   Sin meeple: {no_meeple_count}/{len(results)}")
    
    # Confianza promedio
    avg_confidence = sum(r['confidence']['tile_type'] for r in results) / len(results)
    print(f"\n✨ Confianza promedio: {avg_confidence:.2%}")
    
    # Losetas con baja confianza
    low_confidence = [r for r in results if r['confidence']['tile_type'] < 0.7]
    if low_confidence:
        print(f"\n⚠️ {len(low_confidence)} losetas con baja confianza:")
        for r in low_confidence[:5]:  # Mostrar primeras 5
            print(f"  - {r['image_path']}: {r['tile_letter']} ({r['confidence']['tile_type']:.2%})")
    
    print("\n" + "="*70)
    print("✅ PROCESO COMPLETADO")
    print("="*70)
    print(f"\n📁 Resultados guardados en: board_classification.json")
    print("💡 Abre el archivo JSON para ver los detalles completos\n")


def ejemplo_single_tile():
    """Ejemplo de clasificación de una sola loseta."""
    
    print("\n" + "="*70)
    print("EJEMPLO: CLASIFICACIÓN DE UNA LOSETA")
    print("="*70 + "\n")
    
    from inference import TileClassifier
    
    # Cargar modelo
    model_path = "models/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo no encontrado: {model_path}")
        return
    
    classifier = TileClassifier(model_path)
    
    # Clasificar una loseta de ejemplo
    tile_path = "tiles/tile_001.png"
    
    if not os.path.exists(tile_path):
        print(f"⚠️ Imagen no encontrada: {tile_path}")
        return
    
    print(f"Clasificando: {tile_path}\n")
    
    # Hacer predicción
    result = classifier.predict_single(tile_path)
    
    # Mostrar resultados
    print("📋 RESULTADO:")
    print("-" * 70)
    print(f"  Tipo de loseta: {result['tile_letter']}")
    print(f"  Rotación: {result['rotation']} ({result['rotation'] * 90}°)")
    print(f"  Meeple: {'SÍ ✓' if result['has_meeple'] else 'NO ✗'}")
    
    if result['has_meeple']:
        print(f"  Posición del meeple: {result['meeple_position']}")
    
    print("\n  Confianzas:")
    print(f"    - Tipo: {result['confidence']['tile_type']:.2%}")
    print(f"    - Rotación: {result['confidence']['rotation']:.2%}")
    print(f"    - Meeple: {result['confidence']['meeple_presence']:.2%}")
    
    if result['has_meeple']:
        print(f"    - Posición: {result['confidence']['meeple_position']:.2%}")
    
    print("-" * 70)
    
    # Visualizar
    print("\n¿Quieres ver la visualización? (se abrirá una ventana)")
    print("Presiona cualquier tecla en la ventana para cerrar")
    
    classifier.visualize_prediction(tile_path, result)


def ejemplo_batch_classification():
    """Ejemplo de clasificación en lote."""
    
    print("\n" + "="*70)
    print("EJEMPLO: CLASIFICACIÓN EN LOTE")
    print("="*70 + "\n")
    
    from inference import TileClassifier
    import glob
    
    # Cargar modelo
    model_path = "models/best_model.pth"
    tiles_dir = "tiles"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo no encontrado: {model_path}")
        return
    
    if not os.path.exists(tiles_dir):
        print(f"⚠️ Directorio no encontrado: {tiles_dir}")
        return
    
    # Obtener todas las imágenes
    image_paths = glob.glob(f"{tiles_dir}/*.png") + glob.glob(f"{tiles_dir}/*.jpg")
    
    print(f"Encontradas {len(image_paths)} imágenes en {tiles_dir}/")
    print("Clasificando...\n")
    
    # Crear clasificador
    classifier = TileClassifier(model_path)
    
    # Clasificar en lote
    results = classifier.predict_batch(image_paths, batch_size=32)
    
    # Mostrar muestra de resultados
    print("📊 MUESTRA DE RESULTADOS (primeras 10):")
    print("-" * 70)
    print(f"{'Imagen':<30} {'Tipo':<8} {'Rot':<5} {'Meeple':<8} {'Conf':<8}")
    print("-" * 70)
    
    for i, result in enumerate(results[:10]):
        filename = os.path.basename(result['image_path'])
        tile_type = result['tile_letter']
        rotation = f"{result['rotation']*90}°"
        meeple = "✓" if result['has_meeple'] else "✗"
        confidence = f"{result['confidence']['tile_type']:.1%}"
        
        print(f"{filename:<30} {tile_type:<8} {rotation:<5} {meeple:<8} {confidence:<8}")
    
    if len(results) > 10:
        print(f"... y {len(results)-10} más")
    
    print("-" * 70)
    print(f"\n✓ Clasificación completada para {len(results)} losetas")


def ejemplo_integracion_completa():
    """Ejemplo de integración completa del sistema."""
    
    print("\n" + "="*70)
    print("EJEMPLO: INTEGRACIÓN COMPLETA")
    print("="*70 + "\n")
    
    print("Este es un ejemplo de cómo integrar todo el sistema:\n")
    
    script = '''
import os
import json

# 1. DETECCIÓN: Extraer losetas del tablero
print("🔍 Detectando losetas...")
os.system('python carcassonne.py tablero.jpg')

# 2. CLASIFICACIÓN: Usar IA para identificar cada loseta
print("\\n🤖 Clasificando con IA...")
from inference import classify_tiles_from_detector

results = classify_tiles_from_detector(
    detector_tiles_dir='tiles/',
    model_path='models/best_model.pth',
    output_json='tablero_completo.json'
)

# 3. PROCESAMIENTO: Hacer algo con los resultados
print(f"\\n📊 Procesando {len(results)} losetas...")

# Crear diccionario de tablero
tablero = {}
for result in results:
    # Extraer posición de la loseta del nombre del archivo
    # tile_001_r2_c3.png → row=2, col=3
    import re
    match = re.search(r'_r(-?\\d+)_c(-?\\d+)', result['image_path'])
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        
        tablero[(row, col)] = {
            'tipo': result['tile_letter'],
            'rotacion': result['rotation'],
            'meeple': result['has_meeple'],
            'meeple_pos': result['meeple_position'] if result['has_meeple'] else None
        }

# Guardar tablero estructurado
with open('tablero_estructurado.json', 'w') as f:
    json.dump({str(k): v for k, v in tablero.items()}, f, indent=2)

print("✅ Tablero procesado y guardado en tablero_estructurado.json")

# 4. VISUALIZACIÓN: Crear visualización del tablero
print("\\n🎨 Generando visualización...")
# Aquí podrías crear una visualización del tablero completo
# con matplotlib, pygame, etc.

print("\\n✅ Proceso completado!")
'''
    
    print(script)
    print("\n" + "="*70)
    print("💡 Copia este código y adáptalo a tus necesidades")
    print("="*70 + "\n")


def main():
    """Función principal."""
    
    print("\n🎮 EJEMPLOS DE USO DEL SISTEMA DE IA")
    print("\nSelecciona un ejemplo:")
    print("  1. Flujo completo (detección + clasificación)")
    print("  2. Clasificar una sola loseta")
    print("  3. Clasificación en lote")
    print("  4. Ver código de integración completa")
    print("  5. Ejecutar todos los ejemplos")
    print("  0. Salir")
    
    choice = input("\nOpción (1-5): ").strip()
    
    if choice == '1':
        ejemplo_completo()
    elif choice == '2':
        ejemplo_single_tile()
    elif choice == '3':
        ejemplo_batch_classification()
    elif choice == '4':
        ejemplo_integracion_completa()
    elif choice == '5':
        ejemplo_single_tile()
        ejemplo_batch_classification()
        ejemplo_completo()
        ejemplo_integracion_completa()
    elif choice == '0':
        print("👋 ¡Hasta luego!")
    else:
        print("⚠️ Opción no válida")


if __name__ == "__main__":
    main()
