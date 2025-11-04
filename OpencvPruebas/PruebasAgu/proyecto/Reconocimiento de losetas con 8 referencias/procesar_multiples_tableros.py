#!/usr/bin/env python3
"""
Script para procesar múltiples tableros de Carcassonne y generar dataset de entrenamiento

Este script automatiza todo el proceso para cada foto de tablero:
1. Detectar losetas
2. Auto-anotar (o preparar para anotación manual)
3. Combinar todas las anotaciones
4. Dividir en train/val/test
5. Entrenar modelo

Uso:
    # Procesar múltiples tableros automáticamente
    python procesar_multiples_tableros.py tablero1.jpg tablero2.jpg tablero3.jpg
    
    # O procesar carpeta completa
    python procesar_multiples_tableros.py --dir fotos_tableros/
"""

import sys
import json
import shutil
from pathlib import Path
from typing import List
import argparse
import subprocess


class MultiTableroProcessor:
    """Procesador de múltiples tableros"""
    
    def __init__(self, referencias_dir: str = "referencias"):
        self.referencias_dir = referencias_dir
        self.tableros_procesados = []
        self.all_annotations = []
    
    def procesar_tablero(self, imagen_path: str, tablero_id: int, 
                        auto_annotate: bool = True) -> dict:
        """
        Procesa un tablero completo
        
        Args:
            imagen_path: Ruta a la foto del tablero
            tablero_id: ID único del tablero (1, 2, 3...)
            auto_annotate: Si True, usa auto-anotación; si False, requiere manual
        
        Returns:
            Diccionario con información del procesamiento
        """
        
        print("\n" + "="*60)
        print(f"PROCESANDO TABLERO #{tablero_id}")
        print("="*60)
        print(f"Imagen: {imagen_path}\n")
        
        imagen_path = Path(imagen_path)
        if not imagen_path.exists():
            print(f"✗ Error: Imagen no encontrada: {imagen_path}")
            return None
        
        # Crear directorio para este tablero
        output_dir = Path(f"tablero_{tablero_id:02d}")
        annotations_file = output_dir / f"annotations_{tablero_id:02d}.json"
        # Si ya existe carpeta y anotaciones exitosas, saltar
        if output_dir.exists() and annotations_file.exists():
            try:
                with open(annotations_file, 'r') as f:
                    anns = json.load(f)
                if isinstance(anns, list) and len(anns) > 0:
                    print(f"✓ Tablero ya procesado previamente, saltando: {output_dir}")
                    return {
                        'tablero_id': tablero_id,
                        'imagen': str(imagen_path),
                        'tiles_dir': str(output_dir / 'tiles'),
                        'num_tiles': len(list((output_dir / 'tiles').glob('*.png'))),
                        'num_annotations': len(anns),
                        'annotations_file': str(annotations_file),
                        'status': 'skipped'
                    }
            except Exception as e:
                print(f"⚠️  No se pudo verificar anotaciones previas: {e}")
        output_dir.mkdir(exist_ok=True)
        tiles_dir = output_dir / "tiles"
        
        # PASO 1: Detectar losetas (INTERACTIVO - requiere usuario)
        print(f"[1/3] Detectando losetas...")
        print(f"\n⚠️  ATENCIÓN: El siguiente paso requiere tu interacción")
        print(f"    1. Se abrirá una ventana con la imagen del tablero")
        print(f"    2. Selecciona 8 losetas de referencia distribuidas uniformemente")
        print(f"    3. Presiona ENTER cuando termines")
        print(f"    4. Después presiona 's' para guardar las losetas\n")
        
        input("Presiona ENTER para continuar con la detección interactiva...")
        
        cmd = f'python carcassonne.py "{imagen_path}"'
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode != 0:
            print(f"✗ Error en detección o cancelación del usuario")
            return None
        
        # Verificar que se generó la carpeta tiles/
        tiles_source = Path("tiles")
        if not tiles_source.exists():
            print("✗ No se generó la carpeta tiles/")
            return None
        
        # Mover tiles detectadas al directorio del tablero
        tiles_detectadas = list(tiles_source.glob("*.png"))
        if not tiles_detectadas:
            print("✗ No se detectaron losetas (¿olvidaste presionar 's'?)")
            return None
        
        print(f"✓ {len(tiles_detectadas)} losetas detectadas")
        
        # Crear directorio de destino y mover tiles
        tiles_dir.mkdir(exist_ok=True)
        for tile in tiles_detectadas:
            shutil.move(str(tile), str(tiles_dir / tile.name))
        
        # PASO 2: Anotar losetas
        annotations_file = output_dir / f"annotations_{tablero_id:02d}.json"
        
        if auto_annotate:
            print(f"\n[2/3] Auto-anotando con IA...")
            cmd = f'python auto_annotate.py "{tiles_dir}" "{self.referencias_dir}" --threshold 0.65 --output "{annotations_file}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"⚠️  Auto-anotación falló, requerirá anotación manual")
                print(f"\nEjecuta manualmente:")
                print(f'  python annotation_tool_letters.py "{tiles_dir}" "{self.referencias_dir}"')
                return {
                    'tablero_id': tablero_id,
                    'imagen': str(imagen_path),
                    'tiles_dir': str(tiles_dir),
                    'num_tiles': len(tiles_detectadas),
                    'annotations_file': None,
                    'status': 'pending_annotation'
                }
            
            print(result.stdout)
            
        else:
            print(f"\n[2/3] Anotación manual requerida...")
            print(f"\nEjecuta:")
            print(f'  python annotation_tool_letters.py "{tiles_dir}" "{self.referencias_dir}"')
            print(f"\nLuego guarda como: {annotations_file}")
            
            return {
                'tablero_id': tablero_id,
                'imagen': str(imagen_path),
                'tiles_dir': str(tiles_dir),
                'num_tiles': len(tiles_detectadas),
                'annotations_file': None,
                'status': 'pending_annotation'
            }
        
        # Verificar que se generaron anotaciones
        if not annotations_file.exists():
            print(f"⚠️  Archivo de anotaciones no generado")
            return {
                'tablero_id': tablero_id,
                'imagen': str(imagen_path),
                'tiles_dir': str(tiles_dir),
                'num_tiles': len(tiles_detectadas),
                'annotations_file': None,
                'status': 'failed'
            }
        
        # Cargar anotaciones
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"✓ {len(annotations)} losetas anotadas")
        
        # PASO 3: Agregar metadatos de tablero y ajustar ruta de imagen
        print(f"\n[3/3] Agregando metadatos y corrigiendo rutas...")
        for ann in annotations:
            ann['tablero_id'] = tablero_id
            ann['tablero_imagen'] = str(imagen_path)
            # Ajustar ruta de imagen para que sea relativa al subdirectorio real
            # Si existe 'image_path' o 'filename', actualizarlo
            if 'image_path' in ann:
                ann['image_path'] = str(tiles_dir / Path(ann['image_path']).name).replace('\\', '/').replace('\\', '/')
            elif 'filename' in ann:
                ann['filename'] = str(tiles_dir / Path(ann['filename']).name).replace('\\', '/').replace('\\', '/')
        # Guardar con metadatos y rutas corregidas
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        result = {
            'tablero_id': tablero_id,
            'imagen': str(imagen_path),
            'tiles_dir': str(tiles_dir),
            'num_tiles': len(tiles_detectadas),
            'num_annotations': len(annotations),
            'annotations_file': str(annotations_file),
            'status': 'success'
        }
        
        self.tableros_procesados.append(result)
        self.all_annotations.extend(annotations)
        
        print(f"\n✓ Tablero #{tablero_id} procesado exitosamente")
        
        return result
    
    def procesar_multiples(self, imagenes: List[str], 
                          auto_annotate: bool = True) -> dict:
        """Procesa múltiples tableros"""
        
        print("\n" + "="*60)
        print("PROCESAMIENTO DE MÚLTIPLES TABLEROS")
        print("="*60)
        print(f"\nTableros a procesar: {len(imagenes)}")
        print(f"Modo: {'Auto-anotación' if auto_annotate else 'Anotación manual'}\n")
        
        for i, imagen in enumerate(imagenes, 1):
            self.procesar_tablero(imagen, i, auto_annotate)
        
        return self.generar_reporte()
    
    def generar_reporte(self) -> dict:
        """Genera reporte del procesamiento"""
        
        print("\n" + "="*60)
        print("REPORTE DE PROCESAMIENTO")
        print("="*60)
        
        exitosos = [t for t in self.tableros_procesados if t['status'] == 'success']
        pendientes = [t for t in self.tableros_procesados if t['status'] == 'pending_annotation']
        fallidos = [t for t in self.tableros_procesados if t['status'] == 'failed']
        
        print(f"\nTableros procesados: {len(self.tableros_procesados)}")
        print(f"  ✓ Exitosos: {len(exitosos)}")
        print(f"  ⏳ Pendientes de anotación: {len(pendientes)}")
        print(f"  ✗ Fallidos: {len(fallidos)}")
        
        if exitosos:
            total_tiles = sum(t['num_tiles'] for t in exitosos)
            total_annotations = sum(t['num_annotations'] for t in exitosos)
            print(f"\nTotal de losetas: {total_tiles}")
            print(f"Total anotadas: {total_annotations}")
            print(f"Tasa de anotación: {100*total_annotations/total_tiles:.1f}%")
        
        reporte = {
            'total_tableros': len(self.tableros_procesados),
            'exitosos': len(exitosos),
            'pendientes': len(pendientes),
            'fallidos': len(fallidos),
            'tableros': self.tableros_procesados,
            'total_annotations': len(self.all_annotations)
        }
        
        # Guardar reporte
        with open('reporte_procesamiento.json', 'w') as f:
            json.dump(reporte, f, indent=2)
        
        print("\n✓ Reporte guardado: reporte_procesamiento.json")
        
        return reporte
    
    def combinar_anotaciones(self, output_file: str = "dataset_completo.json"):
        """Combina todas las anotaciones en un solo archivo"""
        
        if not self.all_annotations:
            print("⚠️  No hay anotaciones para combinar")
            return False
        
        print("\n" + "="*60)
        print("COMBINANDO ANOTACIONES")
        print("="*60)
        
        # Estadísticas por tipo
        type_counts = {}
        for ann in self.all_annotations:
            t = ann.get('tile_type', -1)
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"\nTotal de anotaciones: {len(self.all_annotations)}")
        print(f"Tipos únicos: {len(type_counts)}")
        print(f"\nDistribución por tipo:")
        for tipo in sorted(type_counts.keys())[:10]:  # Mostrar primeros 10
            print(f"  Tipo {tipo}: {type_counts[tipo]} losetas")
        
        if len(type_counts) > 10:
            print(f"  ... y {len(type_counts) - 10} tipos más")
        
        # Guardar dataset completo
        with open(output_file, 'w') as f:
            json.dump(self.all_annotations, f, indent=2)
        
        print(f"\n✓ Dataset completo guardado: {output_file}")
        print(f"  Total: {len(self.all_annotations)} losetas")
        
        return True
    
    def dividir_dataset(self, dataset_file: str = "dataset_completo.json"):
        """Divide dataset en train/val/test"""
        
        print("\n" + "="*60)
        print("DIVIDIENDO DATASET")
        print("="*60)
        
        if not Path(dataset_file).exists():
            print(f"✗ Error: {dataset_file} no existe")
            return False
        
        cmd = f'python data-augmentation.py split "{dataset_file}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Error dividiendo dataset: {result.stderr}")
            return False
        
        print(result.stdout)
        print("\n✓ Dataset dividido en train/val/test")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Procesar múltiples tableros de Carcassonne',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Procesar tableros específicos con auto-anotación:
   python procesar_multiples_tableros.py tablero1.jpg tablero2.jpg tablero3.jpg

2. Procesar todos los tableros en una carpeta:
   python procesar_multiples_tableros.py --dir fotos_tableros/

3. Procesar con anotación manual:
   python procesar_multiples_tableros.py --manual tablero1.jpg tablero2.jpg

4. Procesar y entrenar directamente:
   python procesar_multiples_tableros.py --dir fotos/ --train --epochs 100

Workflow completo:
   # 1. Procesar tableros
   python procesar_multiples_tableros.py --dir fotos_tableros/
   
   # 2. (Opcional) Revisar/corregir anotaciones
   python annotation_tool_letters.py tablero_01/tiles/ referencias/
   
   # 3. Combinar y dividir
   python procesar_multiples_tableros.py --combine --split
   
   # 4. Entrenar
   python train_model.py train_annotations.json val_annotations.json
        """
    )
    
    parser.add_argument('imagenes', nargs='*', help='Imágenes de tableros a procesar')
    parser.add_argument('--dir', help='Directorio con imágenes de tableros')
    parser.add_argument('--manual', action='store_true', 
                       help='Usar anotación manual en lugar de auto-anotación')
    parser.add_argument('--referencias', default='referencias',
                       help='Directorio de referencias (default: referencias)')
    parser.add_argument('--combine', action='store_true',
                       help='Solo combinar anotaciones existentes')
    parser.add_argument('--split', action='store_true',
                       help='Solo dividir dataset existente')
    parser.add_argument('--train', action='store_true',
                       help='Entrenar modelo después de procesar')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Epochs para entrenamiento (default: 100)')
    
    args = parser.parse_args()
    
    # Obtener lista de imágenes
    imagenes = []
    
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"✗ Error: Directorio no existe: {args.dir}")
            sys.exit(1)
        
        # Buscar imágenes
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            imagenes.extend(dir_path.glob(ext))
        
        imagenes = [str(img) for img in imagenes]
    
    elif args.imagenes:
        imagenes = args.imagenes
    
    elif not (args.combine or args.split):
        parser.print_help()
        sys.exit(1)
    
    # Procesar
    processor = MultiTableroProcessor(referencias_dir=args.referencias)
    
    if imagenes:
        # Procesar tableros
        processor.procesar_multiples(imagenes, auto_annotate=not args.manual)
        
        # Combinar anotaciones
        if processor.all_annotations:
            processor.combinar_anotaciones()
            processor.dividir_dataset()
    
    elif args.combine:
        # Solo combinar anotaciones existentes
        print("Combinando anotaciones de tableros procesados...")
        annotations = []
        for tablero_dir in sorted(Path('.').glob('tablero_*')):
            ann_file = tablero_dir / f"annotations_{tablero_dir.name.split('_')[1]}.json"
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    annotations.extend(data)
                    print(f"✓ {tablero_dir.name}: {len(data)} anotaciones")
        # Siempre crear el archivo, aunque esté vacío
        with open('dataset_completo.json', 'w') as f:
            json.dump(annotations, f, indent=2)
        if annotations:
            print(f"\n✓ Total: {len(annotations)} anotaciones combinadas")
        else:
            print("⚠️  No se encontraron anotaciones válidas. El archivo dataset_completo.json está vacío.")
    
    if args.split or (imagenes and processor.all_annotations):
        processor.dividir_dataset()
    
    # Entrenar si se solicitó
    if args.train and Path('train_annotations.json').exists():
        print("\n" + "="*60)
        print("ENTRENANDO MODELO")
        print("="*60)
        
        cmd = f'python train_model.py train_annotations.json val_annotations.json {args.epochs}'
        subprocess.run(cmd, shell=True)
    
    print("\n" + "="*60)
    print("✓ PROCESO COMPLETADO")
    print("="*60)
    
    if imagenes and processor.all_annotations:
        print("\nArchivos generados:")
        print("  - tablero_XX/ (directorio por tablero)")
        print("  - dataset_completo.json (todas las anotaciones)")
        print("  - train_annotations.json")
        print("  - val_annotations.json")
        print("  - test_annotations.json")
        print("  - reporte_procesamiento.json")
        
        if args.train:
            print("  - best_carcassonne_model.pth (modelo entrenado)")
        
        print("\nPróximos pasos:")
        if not args.train:
            print("  1. Revisar anotaciones si es necesario")
            print("  2. Entrenar modelo:")
            print("     python train_model.py train_annotations.json val_annotations.json")
        print("  3. Evaluar modelo:")
        print("     python model-evaluation.py best_carcassonne_model.pth test_annotations.json")
        print("  4. Usar en producción:")
        print("     python carcassonne-pipeline.py best_carcassonne_model.pth nueva_foto.jpg")


if __name__ == "__main__":
    main()
