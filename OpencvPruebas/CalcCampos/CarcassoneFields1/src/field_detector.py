"""
Detección y segmentación de campos.
"""
import numpy as np
from scipy import ndimage
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Field:
    """Representa un campo en el tablero."""
    id: int
    pixels: np.ndarray  # Máscara del campo
    meeples: Dict[str, int]  # Conteo de meeples por jugador
    area: int
    

class FieldDetector:
    """Detecta y segmenta campos en el tablero."""
    
    def __init__(self, field_mask: np.ndarray, barrier_mask: np.ndarray):
        """
        Inicializa el detector de campos.
        
        Args:
            field_mask: Máscara de campos (verde)
            barrier_mask: Máscara de barreras (caminos + castillos)
        """
        self.field_mask = field_mask
        self.barrier_mask = barrier_mask
        # Los campos son verdes que NO están en barreras
        self.clean_field_mask = field_mask & ~barrier_mask
    
    def detect_fields(self) -> Tuple[np.ndarray, int]:
        """
        Detecta regiones de campos conectadas.
        
        Returns:
            labeled_array: Array con campos etiquetados
            num_fields: Número de campos detectados
        """
        # Estructura de conectividad (8-conectado)
        structure = np.ones((3, 3), dtype=int)
        
        labeled_array, num_fields = ndimage.label(
            self.clean_field_mask, 
            structure=structure
        )
        
        return labeled_array, num_fields
    
    def count_meeples_in_field(
        self, 
        field_label: int,
        labeled_fields: np.ndarray,
        meeple_masks: Dict[str, np.ndarray]
    ) -> Dict[str, int]:
        """
        Cuenta meeples de cada jugador en un campo.
        
        Args:
            field_label: ID del campo
            labeled_fields: Array de campos etiquetados
            meeple_masks: Diccionario con máscaras de meeples por jugador
            
        Returns:
            Diccionario con conteo de meeples por jugador
        """
        field_pixels = (labeled_fields == field_label)
        counts = {}
        
        for meeple_type, mask in meeple_masks.items():
            # Contar píxeles de meeple dentro del campo
            meeples_in_field = field_pixels & mask
            counts[meeple_type] = np.sum(meeples_in_field)
        
        return counts
    
    def create_fields(
        self, 
        labeled_fields: np.ndarray,
        num_fields: int,
        meeple_masks: Dict[str, np.ndarray]
    ) -> List[Field]:
        """
        Crea objetos Field para cada campo detectado.
        
        Args:
            labeled_fields: Array de campos etiquetados
            num_fields: Número de campos
            meeple_masks: Máscaras de meeples
            
        Returns:
            Lista de objetos Field
        """
        fields = []
        
        for field_id in range(1, num_fields + 1):
            field_pixels = (labeled_fields == field_id)
            area = np.sum(field_pixels)
            
            if area == 0:  # Campo vacío, skip
                continue
            
            meeples = self.count_meeples_in_field(
                field_id, 
                labeled_fields, 
                meeple_masks
            )
            
            field = Field(
                id=field_id,
                pixels=field_pixels,
                meeples=meeples,
                area=area
            )
            
            fields.append(field)
        
        return fields