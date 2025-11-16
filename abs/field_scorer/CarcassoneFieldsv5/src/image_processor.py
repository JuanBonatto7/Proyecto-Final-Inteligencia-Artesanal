"""
Procesamiento de la imagen del tablero.
"""
import numpy as np
import cv2
from typing import Tuple
from config.colors import COLORS, COLOR_TOLERANCE, MEEPLE_TOLERANCE


class ImageProcessor:
    """Procesa la imagen del tablero de Carcassonne."""
    
    def __init__(self, image_path: str):
        """
        Inicializa el procesador de imágenes.
        
        Args:
            image_path: Ruta de la imagen del tablero
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
    
    def create_mask(self, color_name: str) -> np.ndarray:
        """
        Crea una máscara binaria para un color específico.
        
        Args:
            color_name: Nombre del color en COLORS
            
        Returns:
            Máscara binaria donde True indica el color buscado
        """
        target_color = np.array(COLORS[color_name])
        
        # Usar tolerancia específica para meeples
        if 'MEEPLE' in color_name:
            tolerance = MEEPLE_TOLERANCE
        else:
            tolerance = COLOR_TOLERANCE
        
        lower = np.clip(target_color - tolerance, 0, 255)
        upper = np.clip(target_color + tolerance, 0, 255)
        
        mask = cv2.inRange(self.image, lower, upper)
        return mask > 0
    
    def get_combined_barrier_mask(self) -> np.ndarray:
        """
        Crea una máscara combinada de barreras (caminos + castillos).
        
        Returns:
            Máscara binaria de barreras
        """
        road_mask = self.create_mask('ROAD')
        castle_mask = self.create_mask('CASTLE')
        return road_mask | castle_mask

# Integración con el sistema de archivos y ejecución del procesador de imágenes
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Asegurarse de que se proporciona una imagen
    if len(sys.argv) < 2:
        print("Uso: python integrator.py <ruta_imagen>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Verificar si la ruta de la imagen es válida
    image_path = Path(image_path)
    if not image_path.is_file():
        print(f"La ruta de la imagen no es válida: {image_path}")
        sys.exit(1)
    
    # Procesar la imagen
    processor = ImageProcessor(str(image_path))
    barrier_mask = processor.get_combined_barrier_mask()
    
    # Mostrar resultados
    cv2.imshow("Imagen Original", processor.image)
    cv2.imshow("Máscara de Barreras", barrier_mask.astype(np.uint8) * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()