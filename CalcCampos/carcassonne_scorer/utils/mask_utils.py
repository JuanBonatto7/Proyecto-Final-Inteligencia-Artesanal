# ============================================================================
# utils/mask_utils.py
# ============================================================================

from typing import Tuple
import cv2
import numpy as np

from config.settings import Config

class MaskUtils:
    """Utilidades para trabajar con máscaras."""
    
    @staticmethod
    def create_color_mask(image: np.ndarray, 
                          target_color: Tuple[int, int, int],
                          tolerance: int = Config.COLOR_TOLERANCE) -> np.ndarray:
        """
        Crea una máscara binaria para un color específico.
        
        Args:
            image: Imagen BGR
            target_color: Color RGB objetivo
            tolerance: Tolerancia para la detección
            
        Returns:
            Máscara binaria (0 y 255)
        """
        # Convertir color RGB a BGR para OpenCV
        target_bgr = np.array([target_color[2], target_color[1], target_color[0]])
        
        # Crear rangos
        lower = np.clip(target_bgr - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(target_bgr + tolerance, 0, 255).astype(np.uint8)
        
        # Crear máscara
        mask = cv2.inRange(image, lower, upper)
        
        return mask
    
    @staticmethod
    def clean_mask(mask: np.ndarray, kernel_size: int = Config.MORPH_KERNEL_SIZE) -> np.ndarray:
        """
        Limpia una máscara usando operaciones morfológicas.
        
        Args:
            mask: Máscara binaria
            kernel_size: Tamaño del kernel morfológico
            
        Returns:
            Máscara limpia
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Cerrar huecos pequeños
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Abrir para eliminar ruido
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    @staticmethod
    def get_connected_components(mask: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Obtiene componentes conectados de una máscara.
        
        Args:
            mask: Máscara binaria
            
        Returns:
            Tupla (número de componentes, etiquetas)
        """
        num_labels, labels = cv2.connectedComponents(
            mask, 
            connectivity=Config.CONNECTIVITY
        )
        
        return num_labels, labels
    
    @staticmethod
    def masks_touch(mask1: np.ndarray, mask2: np.ndarray) -> bool:
        """
        Determina si dos máscaras se tocan (comparten píxeles adyacentes o superpuestos).
        
        Args:
            mask1: Primera máscara binaria
            mask2: Segunda máscara binaria
            
        Returns:
            True si se tocan
        """
        # Primero verificar superposición directa
        overlap = cv2.bitwise_and(mask1, mask2)
        if np.any(overlap):
            return True
        
        # Luego verificar adyacencia (dilatar una máscara)
        kernel = np.ones((3, 3), np.uint8)
        mask1_dilated = cv2.dilate(mask1, kernel, iterations=1)
        
        # Ver si hay superposición con la máscara dilatada
        intersection = cv2.bitwise_and(mask1_dilated, mask2)
        
        return np.any(intersection)