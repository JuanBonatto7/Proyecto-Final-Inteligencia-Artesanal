"""
Procesamiento de la imagen del tablero.
"""
import numpy as np
import cv2
from typing import Tuple
from config.colors import COLORS, COLOR_TOLERANCE


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
        lower = np.clip(target_color - COLOR_TOLERANCE, 0, 255)
        upper = np.clip(target_color + COLOR_TOLERANCE, 0, 255)
        
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