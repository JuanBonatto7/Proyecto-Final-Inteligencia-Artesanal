#!/usr/bin/env python3
"""
Script para probar el detector con múltiples imágenes
"""

import cv2
import json
from pathlib import Path
from MeepleDetectorSimple import MeepleDetector


def test_multiple_images(images_dir: str = "imagenes_prueba"):
    """
    Procesa todas las imágenes en un directorio
    
    Args:
        images_dir: Directorio con imágenes de prueba
    """
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"❌ Directorio no encontrado: {images_dir}")
        print(f"   Crea la carpeta '{images_dir}' y coloca tus imágenes ahí")
        return
    
    # Buscar todas las imágenes
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(images_path.glob(ext))
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en '{images_dir}'")
        return
    
    print(f"📸 Encontradas {len(image_files)} imágenes")
    print("=" * 60)
    
    # Crear detector
    detector = MeepleDetector()
    
    # Crear carpeta para resultados
    output_dir = Path("resultados")
    output_dir.mkdir(exist_ok=True)
    
    # Procesar cada imagen
    results = []
    stats = {
        'total': len(image_files),
        'with_meeple': 0,
        'blue_count': 0,
        'black_count': 0,
        'unknown_count': 0,
        'positions': [0] * 9
    }
    
    for i, img_file in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] 🔍 {img_file.name}")
        print("-" * 60)
        
        # Detectar
        result = detector.detect_meeple(str(img_file))
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
            continue
        
        # Agregar nombre de archivo al resultado
        result['filename'] = img_file.name
        results.append(result)
        
        # Mostrar resultado
        if result['has_meeple']:
            stats['with_meeple'] += 1
            
            color_emoji = {
                'blue': '🔵',
                'black': '⚫',
                None: '❓'
            }
            
            print(f"  ✅ Meeple detectado!")
            print(f"  {color_emoji.get(result['color'], '❓')} Color: {result['color'] or 'Desconocido'}")
            print(f"  📍 Posición: {result['position'] if result['position'] is not None else 'Fuera del grid'}")
            print(f"  🎯 Confianza: {result['confidence']:.0%}")
            
            # Actualizar estadísticas
            if result['color'] == 'blue':
                stats['blue_count'] += 1
            elif result['color'] == 'black':
                stats['black_count'] += 1
            else:
                stats['unknown_count'] += 1
            
            if result['position'] is not None and 0 <= result['position'] < 9:
                stats['positions'][result['position']] += 1
            
            # Generar visualización
            output_path = output_dir / f"{img_file.stem}_deteccion.jpg"
            detector.visualize_detection(str(img_file), str(output_path))
            print(f"  💾 Guardado: {output_path}")
        else:
            print(f"  ⭕ No se detectó meeple")
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"Total de imágenes: {stats['total']}")
    print(f"Con meeple: {stats['with_meeple']} ({stats['with_meeple']/stats['total']*100:.1f}%)")
    print(f"Sin meeple: {stats['total'] - stats['with_meeple']}")
    print()
    print(f"🔵 Azules: {stats['blue_count']}")
    print(f"⚫ Negros: {stats['black_count']}")
    print(f"❓ Desconocidos: {stats['unknown_count']}")
    print()
    print("📍 Distribución por posición:")
    print("┌───────┬───────┬───────┐")
    for row in range(3):
        line = "│"
        for col in range(3):
            pos = row * 3 + col
            count = stats['positions'][pos]
            line += f"  {count:2d}   │"
        print(line)
        if row < 2:
            print("├───────┼───────┼───────┤")
    print("└───────┴───────┴───────┘")
    
    # Guardar resultados en JSON
    json_path = output_dir / "resultados.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': stats,
            'detections': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {json_path}")
    print(f"📁 Visualizaciones en: {output_dir}/")


if __name__ == "__main__":
    import sys
    
    # Permitir especificar directorio como argumento
    images_dir = sys.argv[1] if len(sys.argv) > 1 else "imagenes_prueba"
    
    test_multiple_images(images_dir)