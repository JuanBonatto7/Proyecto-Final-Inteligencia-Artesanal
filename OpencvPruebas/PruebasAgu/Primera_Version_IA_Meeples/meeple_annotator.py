#!/usr/bin/env python3
"""
Mini software interactivo para anotar meeples en imágenes
Permite al usuario marcar manualmente posiciones de meeples para crear ground truth
"""

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class MeepleAnnotator:
    """Herramienta interactiva para anotar meeples"""

    def __init__(self):
        self.current_image = None
        self.image_display = None
        self.annotations = {}  # image_path -> list of (color, position)
        self.current_meeple_color = 'blue'  # Alterna entre blue/black
        self.temp_annotations = []  # Anotaciones temporales para imagen actual

    def click_callback(self, event, x, y, flags, param):
        """Callback para clics del mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Determinar posición en grid 3x3
            h, w = self.current_image.shape[:2]
            grid_x = int(x / w * 3)
            grid_y = int(y / h * 3)
            position = grid_y * 3 + grid_x

            # Alternar color
            self.current_meeple_color = 'black' if self.current_meeple_color == 'blue' else 'blue'

            # Agregar anotación
            self.temp_annotations.append({
                'color': self.current_meeple_color,
                'position': position,
                'pixel_coords': (x, y)
            })

            print(f"✅ Meeple {self.current_meeple_color} anotado en posición {position} (pixel: {x},{y})")

            # Redibujar
            self.draw_annotations()

    def draw_annotations(self):
        """Dibujar anotaciones en la imagen"""
        display = self.current_image.copy()

        # Dibujar grid 3x3
        h, w = display.shape[:2]
        for i in range(1, 3):
            # Líneas verticales
            cv2.line(display, (int(w*i/3), 0), (int(w*i/3), h), (255, 255, 255), 1)
            # Líneas horizontales
            cv2.line(display, (0, int(h*i/3)), (w, int(h*i/3)), (255, 255, 255), 1)

        # Dibujar anotaciones
        for ann in self.temp_annotations:
            x, y = ann['pixel_coords']
            color = (255, 0, 0) if ann['color'] == 'blue' else (0, 0, 0)
            cv2.circle(display, (x, y), 15, color, 2)
            cv2.putText(display, f"{ann['color'][0].upper()}{ann['position']}",
                       (x-10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mostrar instrucciones
        cv2.putText(display, f"Siguiente color: {self.current_meeple_color}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Click: marcar meeple | ESPACIO: siguiente imagen | Q: salir",
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.image_display = display
        cv2.imshow('Meeple Annotator', display)

    def annotate_image(self, image_path: str) -> Dict:
        """Anotar una imagen específica"""
        # Cargar imagen
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            return {'error': f'No se pudo cargar {image_path}'}

        self.temp_annotations = []
        self.current_meeple_color = 'blue'

        # Configurar ventana
        cv2.namedWindow('Meeple Annotator')
        cv2.setMouseCallback('Meeple Annotator', self.click_callback)

        self.draw_annotations()

        print(f"\n📸 Anotando: {Path(image_path).name}")
        print("Instrucciones:")
        print("- Click izquierdo: marcar meeple (alternará entre azul/negro)")
        print("- ESPACIO: guardar y siguiente imagen")
        print("- Q: salir sin guardar")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q o ESC
                print("❌ Saliendo sin guardar...")
                return {'cancelled': True}

            elif key == ord(' '):  # ESPACIO
                # Guardar anotaciones
                self.annotations[str(image_path)] = self.temp_annotations.copy()
                print(f"✅ Guardadas {len(self.temp_annotations)} anotaciones")
                break

        cv2.destroyWindow('Meeple Annotator')
        return {
            'image_path': str(image_path),
            'annotations': self.temp_annotations.copy()
        }

    def save_annotations(self, output_path: str):
        """Guardar todas las anotaciones"""
        with open(output_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"💾 Anotaciones guardadas en: {output_path}")

    def load_annotations(self, input_path: str):
        """Cargar anotaciones existentes"""
        if Path(input_path).exists():
            with open(input_path, 'r') as f:
                self.annotations = json.load(f)
            print(f"📂 Anotaciones cargadas desde: {input_path}")
        else:
            print(f"⚠️ Archivo de anotaciones no encontrado: {input_path}")

def main():
    """Función principal"""
    annotator = MeepleAnnotator()

    # Cargar anotaciones existentes si las hay
    annotations_file = 'manual_annotations.json'
    annotator.load_annotations(annotations_file)

    # Imágenes para anotar (primeras y últimas de cada letra)
    images_to_annotate = [
        'real_test_images/A20251113_185222.jpg',
        'real_test_images/A20251113_185519.jpg',
        'real_test_images/B20251113_185604.jpg',
        'real_test_images/B20251113_185956.jpg',
        'real_test_images/C20251113_190148.jpg',
        'real_test_images/C20251113_190908.jpg',
        # Agregar más si el usuario quiere...
    ]

    print("🎯 Mini Software de Anotación de Meeples")
    print("=" * 50)
    print("Este software te permitirá marcar manualmente dónde están los meeples")
    print("para crear datos ground truth y mejorar el detector automático.")
    print()

    completed = 0
    for img_path in images_to_annotate:
        if str(img_path) in annotator.annotations:
            print(f"⏭️ {Path(img_path).name} ya anotada, saltando...")
            completed += 1
            continue

        result = annotator.annotate_image(img_path)
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue
        elif result.get('cancelled'):
            break

        completed += 1

        # Preguntar si continuar
        if completed < len(images_to_annotate):
            print(f"Progreso: {completed}/{len(images_to_annotate)} imágenes anotadas")
            response = input("¿Continuar con la siguiente imagen? (s/n): ").lower()
            if response != 's':
                break

    # Guardar anotaciones
    if annotator.annotations:
        annotator.save_annotations(annotations_file)
        print(f"\n✅ Sesión completada. Anotaste {len(annotator.annotations)} imágenes.")
        print("Ahora podemos usar estas anotaciones para mejorar el detector automático.")
    else:
        print("\n⚠️ No se guardaron anotaciones.")

if __name__ == "__main__":
    main()