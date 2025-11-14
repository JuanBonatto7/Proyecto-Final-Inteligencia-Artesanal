#!/usr/bin/env python3
"""
Script para agregar nuevas imágenes de referencia al dataset organizado
"""
import os
import shutil
from pathlib import Path
import argparse

def add_reference_images(source_folder, dest_folder="referencias_organizadas"):
    """Agrega nuevas imágenes de referencia desde una carpeta fuente"""

    source_path = Path(source_folder)
    dest_path = Path(dest_folder)

    if not source_path.exists():
        print(f"Error: Carpeta fuente '{source_folder}' no existe")
        return False

    if not dest_path.exists():
        print(f"Error: Carpeta destino '{dest_folder}' no existe")
        return False

    print(f"Agregando imágenes desde: {source_folder}")
    print(f"A la estructura organizada en: {dest_folder}")

    added_count = 0

    # Procesar todas las imágenes PNG en la carpeta fuente
    for img_file in source_path.glob("*.png"):
        filename = img_file.stem.upper()  # Convertir a mayúsculas

        # Intentar determinar la letra desde el nombre del archivo
        # Asumir que el nombre comienza con la letra (ej: A_tile_001.png -> A)
        if len(filename) > 0 and filename[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            letter = filename[0]
            dest_letter_folder = dest_path / letter

            if dest_letter_folder.exists():
                # Crear nombre único
                counter = 1
                dest_file = dest_letter_folder / f"{letter}_ref_{counter:03d}.png"
                while dest_file.exists():
                    counter += 1
                    dest_file = dest_letter_folder / f"{letter}_ref_{counter:03d}.png"

                # Copiar la imagen
                shutil.copy2(img_file, dest_file)
                print(f"  Agregado: {img_file.name} -> {letter}/{dest_file.name}")
                added_count += 1
            else:
                print(f"  [SKIP] Carpeta para {letter} no existe")
        else:
            print(f"  [SKIP] No se pudo determinar letra para {img_file.name}")

    print(f"\nProceso completado: {added_count} imágenes agregadas")
    return added_count > 0

def show_current_stats(dest_folder="referencias_organizadas"):
    """Muestra estadísticas actuales del dataset"""

    dest_path = Path(dest_folder)
    if not dest_path.exists():
        print(f"Carpeta '{dest_folder}' no existe")
        return

    print(f"\nEstadísticas actuales del dataset en '{dest_folder}':")
    print("-" * 50)

    total_images = 0
    for letter in sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        letter_folder = dest_path / letter
        if letter_folder.exists():
            img_count = len(list(letter_folder.glob("*.png")))
            print("2d")
            total_images += img_count
        else:
            print("2d")

    print("-" * 50)
    print(f"Total imágenes: {total_images}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agregar imágenes de referencia al dataset organizado")
    parser.add_argument("source_folder", help="Carpeta con las nuevas imágenes a agregar")
    parser.add_argument("--dest", default="referencias_organizadas", help="Carpeta destino organizada")

    args = parser.parse_args()

    # Mostrar estadísticas antes
    show_current_stats(args.dest)

    # Agregar imágenes
    success = add_reference_images(args.source_folder, args.dest)

    # Mostrar estadísticas después
    if success:
        show_current_stats(args.dest)

    print("\nRecuerda re-entrenar el modelo CNN después de agregar nuevas imágenes:")
    print("python train_cnn_multi.py")