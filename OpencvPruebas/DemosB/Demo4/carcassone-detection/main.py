"""
Punto de entrada principal del sistema de detección de Carcassonne.
"""

import sys
import cv2
from pathlib import Path

from pipeline.pipeline import Pipeline
from steps import ResizeStep
from config import Config


def create_detection_pipeline() -> Pipeline:
    pipeline = Pipeline("Carcassonne Board Detector")
    pipeline.add_step(ResizeStep())
    return pipeline


def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen desde {path}")
        return None
    return image


def main():
    print("\n" + "="*60)
    print("  🎮 SISTEMA DE DETECCIÓN DE TABLERO DE CARCASSONNE")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n❌ Error: Debe proporcionar la ruta a una imagen")
        print(f"\nUso: python {sys.argv[0]} <ruta_imagen>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"\n❌ Error: El archivo no existe: {image_path}")
        sys.exit(1)
    
    print(f"\n📷 Cargando imagen: {image_path}")
    image = load_image(image_path)
    
    if image is None:
        sys.exit(1)
    
    print(f"✓ Imagen cargada: {image.shape[1]}x{image.shape[0]} píxeles")
    
    pipeline = create_detection_pipeline()
    
    inputs = {'img': image}
    
    results = pipeline.run(inputs, visualize=True)
    
    print("\n✅ Procesamiento completado!\n")


if __name__ == "__main__":
    main()