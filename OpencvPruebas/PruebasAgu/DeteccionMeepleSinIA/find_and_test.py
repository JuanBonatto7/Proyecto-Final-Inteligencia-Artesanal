    #!/usr/bin/env python3
"""
Script auxiliar para encontrar y probar imágenes
"""

import os
import sys
from pathlib import Path
from MeepleDetectorSimple import MeepleDetector


def find_images_in_directory(directory="."):
    """Encuentra todas las imágenes en un directorio"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    images = []
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in image_extensions):
            images.append(os.path.join(directory, file))
    
    return sorted(images)


def test_single_image(image_path):
    """Prueba con una sola imagen"""
    detector = MeepleDetector()
    
    print(f"\n🔍 Analizando: {os.path.basename(image_path)}")
    print("=" * 60)
    
    result = detector.detect_meeple(image_path)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    # Mostrar resultados
    print(f"¿Hay meeple?: {'✅ SÍ' if result['has_meeple'] else '⭕ NO'}")
    
    if result['has_meeple']:
        color_emoji = {'blue': '🔵', 'black': '⚫', None: '❓'}
        print(f"{color_emoji.get(result['color'], '❓')} Color: {result['color'] or 'Desconocido'}")
        print(f"📍 Posición: {result['position'] if result['position'] is not None else 'Fuera del grid'}")
        print(f"🎯 Confianza: {result['confidence']:.0%}")
        
        if result['circle']:
            x, y, r = result['circle']
            print(f"⭕ Círculo: centro=({x},{y}), radio={r}px")
        
        # Generar visualización
        output_path = f"deteccion_{Path(image_path).stem}.jpg"
        detector.visualize_detection(image_path, output_path)
        print(f"💾 Visualización guardada: {output_path}")
    
    return True


def interactive_mode():
    """Modo interactivo para seleccionar imagen"""
    print("🔍 DETECTOR DE MEEPLES - Modo Interactivo")
    print("=" * 60)
    
    # Buscar imágenes
    images = find_images_in_directory()
    
    if not images:
        print("❌ No se encontraron imágenes en el directorio actual.")
        print("\n💡 Coloca tus imágenes en esta carpeta y ejecuta nuevamente.")
        return
    
    print(f"\n📸 Encontradas {len(images)} imágenes:\n")
    
    for i, img in enumerate(images, 1):
        print(f"  {i}. {os.path.basename(img)}")
    
    print(f"\n  0. Procesar todas")
    print(f"  Q. Salir")
    
    while True:
        choice = input("\n👉 Selecciona una imagen (número): ").strip().lower()
        
        if choice == 'q':
            print("👋 ¡Hasta luego!")
            break
        
        if choice == '0':
            # Procesar todas
            print("\n🚀 Procesando todas las imágenes...")
            for img in images:
                test_single_image(img)
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                test_single_image(images[idx])
                
                # Preguntar si continuar
                cont = input("\n¿Analizar otra imagen? (s/n): ").strip().lower()
                if cont != 's':
                    break
            else:
                print("❌ Número inválido")
        except ValueError:
            print("❌ Entrada inválida")


def main():
    if len(sys.argv) > 1:
        # Modo de línea de comandos
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"❌ Archivo no encontrado: {image_path}")
            print("\n📁 Imágenes disponibles:")
            for img in find_images_in_directory():
                print(f"   - {os.path.basename(img)}")
            return
        
        test_single_image(image_path)
    else:
        # Modo interactivo
        interactive_mode()


if __name__ == "__main__":
    main()