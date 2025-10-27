"""
Herramienta de debug para ver qué colores detecta el programa.
"""
import sys
import cv2
import numpy as np
from src.image_processor import ImageProcessor
from config.colors import COLORS


def show_color_masks(image_path):
    """Muestra las máscaras de cada color."""
    
    processor = ImageProcessor(image_path)
    
    print("Dimensiones de imagen:", processor.image.shape)
    print("\nColores configurados:")
    for name, color in COLORS.items():
        print(f"  {name}: RGB{color}")
    
    print("\nCreando máscaras...")
    
    # Mostrar imagen original
    cv2.imshow('Original', cv2.cvtColor(processor.image, cv2.COLOR_RGB2BGR))
    
    # Crear y mostrar máscara para cada color
    for color_name in COLORS.keys():
        mask = processor.create_mask(color_name)
        pixel_count = np.sum(mask)
        
        print(f"\n{color_name}:")
        print(f"  Pixeles detectados: {pixel_count}")
        print(f"  Porcentaje: {pixel_count / mask.size * 100:.2f}%")
        
        # Convertir máscara a imagen visible
        mask_img = np.zeros_like(processor.image)
        mask_img[mask] = COLORS[color_name]
        
        cv2.imshow(f'{color_name}', cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
    
    # Mostrar máscara de barreras
    barrier_mask = processor.get_combined_barrier_mask()
    barrier_count = np.sum(barrier_mask)
    print(f"\nBARRERAS (Caminos + Castillos):")
    print(f"  Pixeles detectados: {barrier_count}")
    
    barrier_img = np.zeros_like(processor.image)
    barrier_img[barrier_mask] = (255, 255, 0)  # Amarillo
    cv2.imshow('Barreras', cv2.cvtColor(barrier_img, cv2.COLOR_RGB2BGR))
    
    print("\nPresiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python debug_colors.py <ruta_imagen>")
        sys.exit(1)
    
    show_color_masks(sys.argv[1])