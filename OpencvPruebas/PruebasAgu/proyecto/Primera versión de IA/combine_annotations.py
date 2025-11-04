#!/usr/bin/env python3
"""
Combina múltiples archivos de anotaciones en uno solo
Útil para mezclar anotaciones automáticas con manuales
"""

import json
import sys
from pathlib import Path
from typing import List, Dict


def load_annotations(file_path: str) -> List[Dict]:
    """Carga anotaciones desde un archivo JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def combine_annotations(*annotation_files: str, 
                       priority: str = 'last',
                       remove_duplicates: bool = True) -> List[Dict]:
    """
    Combina múltiples archivos de anotaciones
    
    Args:
        annotation_files: Rutas a los archivos de anotaciones
        priority: 'first' o 'last' - qué archivo tiene prioridad en duplicados
        remove_duplicates: Si True, mantiene solo una anotación por imagen
    
    Returns:
        Lista combinada de anotaciones
    """
    all_annotations = []
    seen_images = {}
    
    # Cargar todos los archivos
    for file_path in annotation_files:
        if not Path(file_path).exists():
            print(f"⚠️  Advertencia: {file_path} no existe, saltando...")
            continue
        
        try:
            annotations = load_annotations(file_path)
            print(f"✓ Cargadas {len(annotations)} anotaciones de {file_path}")
            
            for ann in annotations:
                image_path = ann['image_path']
                
                if remove_duplicates:
                    if priority == 'last':
                        # Última versión tiene prioridad
                        seen_images[image_path] = ann
                    elif priority == 'first':
                        # Primera versión tiene prioridad
                        if image_path not in seen_images:
                            seen_images[image_path] = ann
                else:
                    # No eliminar duplicados, agregar todo
                    all_annotations.append(ann)
        
        except Exception as e:
            print(f"✗ Error cargando {file_path}: {e}")
            continue
    
    if remove_duplicates:
        all_annotations = list(seen_images.values())
    
    return all_annotations


def print_statistics(annotations: List[Dict]):
    """Imprime estadísticas sobre las anotaciones"""
    print(f"\n{'='*60}")
    print("ESTADÍSTICAS DE ANOTACIONES COMBINADAS")
    print(f"{'='*60}")
    
    print(f"Total de anotaciones: {len(annotations)}")
    
    # Contar por tipo
    type_counts = {}
    for ann in annotations:
        tile_letter = ann.get('tile_letter', 'DESCONOCIDO')
        type_counts[tile_letter] = type_counts.get(tile_letter, 0) + 1
    
    print(f"\nDistribución por tipo:")
    for letter in sorted(type_counts.keys()):
        print(f"  {letter}: {type_counts[letter]}")
    
    # Contar auto vs manual
    auto_count = sum(1 for ann in annotations if ann.get('auto_annotated', False))
    manual_count = len(annotations) - auto_count
    
    print(f"\nOrigen:")
    print(f"  Automáticas: {auto_count}")
    print(f"  Manuales: {manual_count}")
    
    # Confianza promedio (solo auto-anotadas)
    auto_annotations = [ann for ann in annotations if 'confidence' in ann]
    if auto_annotations:
        avg_confidence = sum(ann['confidence'] for ann in auto_annotations) / len(auto_annotations)
        print(f"\nConfianza promedio (auto): {avg_confidence:.2%}")
    
    print(f"{'='*60}\n")


def save_annotations(annotations: List[Dict], output_file: str):
    """Guarda las anotaciones combinadas"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Anotaciones guardadas en: {output_file}")


def main():
    if len(sys.argv) < 3:
        print("Uso: python combine_annotations.py <archivo1.json> <archivo2.json> [archivo3.json ...] <output.json>")
        print("\nEjemplo:")
        print("  python combine_annotations.py auto_annotations.json manual_annotations.json combined.json")
        print("\nOpciones:")
        print("  --priority first  : Primera anotación tiene prioridad en duplicados")
        print("  --priority last   : Última anotación tiene prioridad (default)")
        print("  --keep-duplicates : No eliminar duplicados")
        sys.exit(1)
    
    # Parsear argumentos
    args = sys.argv[1:]
    priority = 'last'
    remove_duplicates = True
    
    # Procesar flags
    while args and args[0].startswith('--'):
        flag = args.pop(0)
        if flag == '--priority':
            priority = args.pop(0)
        elif flag == '--keep-duplicates':
            remove_duplicates = False
    
    if len(args) < 2:
        print("✗ Error: Necesitas al menos un archivo de entrada y uno de salida")
        sys.exit(1)
    
    # Último argumento es el output
    output_file = args[-1]
    input_files = args[:-1]
    
    print(f"{'='*60}")
    print("COMBINADOR DE ANOTACIONES")
    print(f"{'='*60}\n")
    
    print(f"Archivos de entrada: {len(input_files)}")
    for f in input_files:
        print(f"  - {f}")
    print(f"\nArchivo de salida: {output_file}")
    print(f"Prioridad en duplicados: {priority}")
    print(f"Eliminar duplicados: {remove_duplicates}\n")
    
    # Combinar anotaciones
    combined = combine_annotations(*input_files, 
                                   priority=priority,
                                   remove_duplicates=remove_duplicates)
    
    if not combined:
        print("✗ No se pudo combinar ninguna anotación")
        sys.exit(1)
    
    # Mostrar estadísticas
    print_statistics(combined)
    
    # Guardar resultado
    save_annotations(combined, output_file)
    
    print("\n✓ Proceso completado exitosamente")


if __name__ == "__main__":
    main()
