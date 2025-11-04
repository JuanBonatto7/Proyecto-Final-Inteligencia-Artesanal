#!/usr/bin/env python3
"""
Script interactivo para revisión/corrección de predicciones de la IA (active learning)

- Muestra cada loseta y la predicción de la IA
- Permite aceptar o corregir el tipo de loseta
- Guarda solo los ejemplos corregidos en corrections.json

Uso:
    python active_learning_review.py --images_dir tablero_01/tiles/ --model best_carcassonne_model.pth --output corrections.json
"""
import argparse
import json
import os
from pathlib import Path
import cv2

# Simulación de predicción IA (reemplazar por tu modelo real)
def predict_tile_type(model_path, image_path):
    # Aquí deberías cargar tu modelo y predecir el tipo de loseta
    # Por ahora, simulamos con un tipo aleatorio
    import random
    tipos = list("ABCDEFGHIJKLMNOPQRSTUVWX ")  # 24 tipos + blanco
    return random.choice(tipos)

def main():
    parser = argparse.ArgumentParser(description="Revisión/corrección de predicciones de la IA (active learning)")
    parser.add_argument('--images_dir', required=True, help='Directorio con imágenes de losetas')
    parser.add_argument('--model', required=True, help='Modelo entrenado .pth')
    parser.add_argument('--output', default='corrections.json', help='Archivo para guardar correcciones')
    args = parser.parse_args()

    images = sorted([str(p) for p in Path(args.images_dir).glob('*.png')])
    if not images:
        print(f"No se encontraron imágenes en {args.images_dir}")
        return

    corrections = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer {img_path}")
            continue

        # Predicción de la IA
        pred = predict_tile_type(args.model, img_path)

        # Mostrar imagen y predicción
        display = img.copy()
        cv2.putText(display, f"Prediccion IA: {pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Revisar loseta", display)
        print(f"Imagen: {img_path}")
        print(f"Prediccion IA: {pred}")
        print("[Enter]=Aceptar, [Letra]=Corregir tipo, [q]=Salir")
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == 13 or key == 10:  # Enter
            # Aceptar predicción
            continue
        else:
            # Corregir (usuario ingresa letra)
            tipo = chr(key).upper()
            print(f"Corregido a: {tipo}")
            corrections.append({
                'image_path': img_path,
                'predicted': pred,
                'corrected': tipo
            })
    cv2.destroyAllWindows()
    if corrections:
        with open(args.output, 'w') as f:
            json.dump(corrections, f, indent=2)
        print(f"Correcciones guardadas en {args.output}")
    else:
        print("No hubo correcciones. ¡La IA lo hizo perfecto o no revisaste nada!")

if __name__ == "__main__":
    main()
