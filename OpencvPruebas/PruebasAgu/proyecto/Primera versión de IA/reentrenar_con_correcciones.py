#!/usr/bin/env python3
"""
Script para automatizar el reentrenamiento de la IA usando correcciones de active learning.

- Lee corrections_full.json (o similar)
- Actualiza/crea un nuevo archivo de anotaciones para entrenamiento
- Llama a train_model.py con el dataset ampliado

Uso:
    python reentrenar_con_correcciones.py --corrections corrections_full.json --base_annotations train_annotations.json --val_annotations val_annotations.json --epochs 50
"""
import argparse
import json
import shutil
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Reentrenar IA con correcciones de active learning")
    parser.add_argument('--corrections', required=True, help='Archivo de correcciones (corrections_full.json)')
    parser.add_argument('--base_annotations', required=True, help='Archivo base de anotaciones (train_annotations.json)')
    parser.add_argument('--val_annotations', required=True, help='Archivo de validación (val_annotations.json)')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs para reentrenamiento')
    parser.add_argument('--output', default='train_annotations_corr.json', help='Archivo de salida para anotaciones corregidas')
    args = parser.parse_args()

    # Cargar anotaciones base
    with open(args.base_annotations, 'r') as f:
        base_anns = json.load(f)

    # Cargar correcciones
    with open(args.corrections, 'r') as f:
        corrections = json.load(f)

    # Indexar base por ruta de imagen
    ann_by_path = {ann['image_path']: ann for ann in base_anns}

    # Aplicar correcciones
    for corr in corrections:
        path = corr['image_path']
        if path in ann_by_path:
            ann = ann_by_path[path]
            ann['tile_type'] = corr['corrected']['tipo']
            ann['rotation'] = corr['corrected']['rotacion']
            ann['has_meeple'] = corr['corrected']['tiene_ficha']
            ann['meeple_position'] = corr['corrected']['pos_ficha']
        else:
            # Si no está en base, agregar como nuevo ejemplo
            ann_by_path[path] = {
                'image_path': path,
                'tile_type': corr['corrected']['tipo'],
                'rotation': corr['corrected']['rotacion'],
                'has_meeple': corr['corrected']['tiene_ficha'],
                'meeple_position': corr['corrected']['pos_ficha']
            }
    # Guardar nuevo archivo de anotaciones
    new_anns = list(ann_by_path.values())
    with open(args.output, 'w') as f:
        json.dump(new_anns, f, indent=2)
    print(f"Anotaciones corregidas guardadas en {args.output}")

    # Llamar entrenamiento
    print("Entrenando modelo con correcciones...")
    cmd = f'python train_model.py {args.output} {args.val_annotations} {args.epochs}'
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()
