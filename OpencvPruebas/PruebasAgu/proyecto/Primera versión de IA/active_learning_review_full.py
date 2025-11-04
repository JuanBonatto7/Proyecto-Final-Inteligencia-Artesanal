#!/usr/bin/env python3
"""
Active Learning: Revisión/corrección de predicciones multi-tarea (tipo, rotación, ficha)

- Muestra cada loseta y la predicción de la IA (tipo, rotación, ficha)
- Permite aceptar o corregir cada campo
- Guarda solo los ejemplos corregidos en corrections_full.json

Uso:
    python active_learning_review_full.py --images_dir tablero_01/tiles/ --model best_carcassonne_model.pth --output corrections_full.json
"""
import argparse
import json
from pathlib import Path
import cv2

# Simulación de predicción IA multi-tarea (reemplazar por tu modelo real)
def predict_tile(model_path, image_path):
    # Aquí deberías cargar tu modelo y predecir tipo, rotación y ficha
    # Simulación:
    import random
    tipos = list("ABCDEFGHIJKLMNOPQRSTUVWX ")  # 24 tipos + blanco
    tipo = random.choice(tipos)
    rotacion = random.randint(0, 3)
    tiene_ficha = random.choice([True, False])
    pos_ficha = random.randint(0, 8) if tiene_ficha else None
    return {
        'tipo': tipo,
        'rotacion': rotacion,
        'tiene_ficha': tiene_ficha,
        'pos_ficha': pos_ficha
    }

def main():
    parser = argparse.ArgumentParser(description="Active Learning: revisión/corrección multi-tarea")
    parser.add_argument('--images_dir', required=True, help='Directorio con imágenes de losetas')
    parser.add_argument('--model', required=True, help='Modelo entrenado .pth')
    parser.add_argument('--output', default='corrections_full.json', help='Archivo para guardar correcciones')
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
        pred = predict_tile(args.model, img_path)

        # Mostrar imagen y predicción
        display = img.copy()
        cv2.putText(display, f"Tipo: {pred['tipo']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(display, f"Rot: {pred['rotacion']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(display, f"Ficha: {pred['tiene_ficha']}" + (f" ({pred['pos_ficha']})" if pred['tiene_ficha'] else ""), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.imshow("Revisar loseta", display)
        print(f"Imagen: {img_path}")
        print(f"Predicción IA: tipo={pred['tipo']}, rotación={pred['rotacion']}, ficha={pred['tiene_ficha']}, pos_ficha={pred['pos_ficha']}")
        print("[Enter]=Aceptar, [t]=corregir tipo, [r]=corregir rotación, [f]=corregir ficha, [q]=Salir")
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == 13 or key == 10:  # Enter
            continue  # Aceptar predicción
        else:
            corr = pred.copy()
            if key == ord('t'):
                tipo = input("Tipo correcto (A-X o espacio): ").strip().upper()
                if tipo:
                    corr['tipo'] = tipo
            if key == ord('r'):
                rot = input("Rotación correcta (0-3): ").strip()
                if rot.isdigit() and 0 <= int(rot) <= 3:
                    corr['rotacion'] = int(rot)
            if key == ord('f'):
                tiene = input("¿Tiene ficha? (s/n): ").strip().lower()
                if tiene == 's':
                    corr['tiene_ficha'] = True
                    pos = input("Posición ficha (0-8): ").strip()
                    if pos.isdigit() and 0 <= int(pos) <= 8:
                        corr['pos_ficha'] = int(pos)
                    else:
                        corr['pos_ficha'] = 0
                else:
                    corr['tiene_ficha'] = False
                    corr['pos_ficha'] = None
            corrections.append({
                'image_path': img_path,
                'predicted': pred,
                'corrected': corr
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
