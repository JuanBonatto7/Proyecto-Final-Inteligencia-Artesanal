# ============================================================================
# processors/image_processor.py
# ============================================================================

import cv2
import numpy as np
from typing import Dict

from CalcCampos.carcassonne_scorer.config.settings import Config
from CalcCampos.carcassonne_scorer.utils.mask_utils import MaskUtils

class ImageProcessor:
    """Procesador de imágenes del tablero."""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = None
        self.masks: Dict[str, np.ndarray] = {}
    
    def load_image(self) -> np.ndarray:
        """Carga la imagen del tablero."""
        self.image = cv2.imread(self.image_path)
        
        if self.image is None:
            raise ValueError(f"No se pudo cargar la imagen: {self.image_path}")
        
        # Verificar dimensiones
        if self.image.shape[:2] != Config.IMAGE_SIZE:
            print(f"Advertencia: La imagen tiene dimensiones {self.image.shape[:2]}, "
                  f"se esperaba {Config.IMAGE_SIZE}")
        
        return self.image
    
    def create_all_masks(self):
        """Crea máscaras para todos los elementos del juego."""
        if self.image is None:
            raise ValueError("Primero debe cargar la imagen con load_image()")
        
        for name, color in Config.COLORS.items():
            mask = MaskUtils.create_color_mask(self.image, color)
            mask = MaskUtils.clean_mask(mask)
            self.masks[name] = mask
    
    def get_mask(self, name: str) -> np.ndarray:
        """Obtiene una máscara específica."""
        return self.masks.get(name, np.zeros(Config.IMAGE_SIZE, dtype=np.uint8))