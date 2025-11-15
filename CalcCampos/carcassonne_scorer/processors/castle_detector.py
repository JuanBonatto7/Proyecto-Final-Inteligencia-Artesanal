# ============================================================================
# processors/castle_detector.py
# ============================================================================

from dataclasses import Field
import numpy as np
from CalcCampos.carcassonne_scorer.processors.image_processor import ImageProcessor
from CalcCampos.carcassonne_scorer.utils.mask_utils import MaskUtils


class CastleDetector:
    """Detector de castillos cerrados."""
    
    def __init__(self, image_processor: ImageProcessor):
        self.processor = image_processor
    
    def is_castle_closed(self, castle_mask: np.ndarray, empty_mask: np.ndarray) -> bool:
        """
        Determina si un castillo está cerrado.
        
        Un castillo está CERRADO si NO toca ningún píxel blanco (loseta vacía).
        Un castillo está ABIERTO si toca al menos un píxel blanco.
        
        Args:
            castle_mask: Máscara del castillo individual
            empty_mask: Máscara de losetas vacías (píxeles blancos)
            
        Returns:
            True si el castillo está cerrado
        """
        # Si el castillo NO toca ningún píxel blanco, está cerrado
        return not MaskUtils.masks_touch(castle_mask, empty_mask)
    
    def count_closed_castles_touching_field(self, field: Field) -> int:
        """
        Cuenta cuántos castillos CERRADOS tocan un campo.
        
        Para cada castillo:
        1. Verificar si está cerrado (NO toca píxeles blancos)
        2. Si está cerrado, verificar si toca el campo
        3. Contar solo los castillos cerrados que tocan el campo
        
        Args:
            field: Campo a analizar
            
        Returns:
            Número de castillos cerrados que tocan el campo
        """
        castle_mask = self.processor.get_mask('CASTLE')
        empty_mask = self.processor.get_mask('EMPTY')
        
        # Obtener componentes de castillos
        num_castles, castle_labels = MaskUtils.get_connected_components(castle_mask)
        
        closed_touching_count = 0
        
        for castle_id in range(1, num_castles):
            # Máscara de este castillo específico
            single_castle = (castle_labels == castle_id).astype(np.uint8) * 255
            
            # Primero verificar si el castillo está CERRADO
            if not self.is_castle_closed(single_castle, empty_mask):
                continue  # Este castillo está ABIERTO, no cuenta
            
            # El castillo está CERRADO, ahora verificar si toca el campo
            if MaskUtils.masks_touch(single_castle, field.mask):
                closed_touching_count += 1
        
        return closed_touching_count
