# ============================================================================
# processors/field_detector.py
# ============================================================================

from typing import List
import numpy as np

from CalcCampos.carcassonne_scorer.models.field import Field
from CalcCampos.carcassonne_scorer.processors.image_processor import ImageProcessor
from CalcCampos.carcassonne_scorer.utils.mask_utils import MaskUtils
from config.settings import Config

class FieldDetector:
    """Detector de campos en el tablero."""
    
    def __init__(self, image_processor: ImageProcessor):
        self.processor = image_processor
    
    def detect_fields(self) -> List[Field]:
        """
        Detecta todos los campos en el tablero.
        
        Returns:
            Lista de campos detectados
        """
        field_mask = self.processor.get_mask('FIELD')
        
        # Obtener componentes conectados
        num_labels, labels = MaskUtils.get_connected_components(field_mask)
        
        fields = []
        
        # Iterar sobre cada componente (empezando en 1, 0 es el fondo)
        for label_id in range(1, num_labels):
            # Crear máscara para este campo
            mask = (labels == label_id).astype(np.uint8) * 255
            
            # Calcular área
            area = np.sum(mask > 0)
            
            # Filtrar campos muy pequeños
            if area < Config.MIN_FIELD_AREA:
                continue
            
            field = Field(
                id=label_id,
                mask=mask,
                area=area
            )
            
            fields.append(field)
        
        return fields
    
    def count_meeples_in_field(self, field: Field):
        """
        Cuenta los meeples de cada jugador en un campo.
        
        Un meeple está en un campo si:
        1. Toca al menos un píxel del campo
        2. NO toca ningún píxel de camino
        
        Args:
            field: Campo a analizar
        """
        meeple1_mask = self.processor.get_mask('MEEPLE_1')
        meeple2_mask = self.processor.get_mask('MEEPLE_2')
        road_mask = self.processor.get_mask('ROAD')
        
        # Para cada tipo de meeple
        for player, meeple_mask in [(1, meeple1_mask), (2, meeple2_mask)]:
            # Obtener meeples de este jugador
            num_meeples, meeple_labels = MaskUtils.get_connected_components(meeple_mask)
            
            count = 0
            
            for meeple_id in range(1, num_meeples):
                # Máscara de este meeple específico
                single_meeple = (meeple_labels == meeple_id).astype(np.uint8) * 255
                
                # Verificar área mínima
                if np.sum(single_meeple) < Config.MIN_MEEPLE_AREA:
                    continue
                
                # Verificar si toca el campo
                if MaskUtils.masks_touch(single_meeple, field.mask):
                    # Verificar que NO toque un camino
                    if not MaskUtils.masks_touch(single_meeple, road_mask):
                        count += 1
            
            if player == 1:
                field.meeples_p1 = count
            else:
                field.meeples_p2 = count