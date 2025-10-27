"""
Detección y segmentación de campos (CORREGIDO).
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
    
    def expand_barriers(self, iterations=3):
        """
        Expande las barreras para separar mejor los campos.
        Esto ayuda a que los caminos delgados separen correctamente.
        
        Args:
            iterations: Número de iteraciones de dilatación
            
        Returns:
            Máscara de barreras expandida
        """
        # Crear kernel para dilatación
        kernel = np.ones((3, 3), dtype=np.uint8)
        
        # Dilatar las barreras
        expanded = self.barrier_mask.astype(np.uint8)
        expanded = ndimage.binary_dilation(expanded, structure=kernel, iterations=iterations)
        
        return expanded
    
    def detect_fields(self, expand_barriers_iterations=3, min_area=50) -> Tuple[np.ndarray, int]:
        """
        Detecta regiones de campos conectadas.
        
        Args:
            expand_barriers_iterations: Cuánto expandir las barreras (más = mejor separación)
            min_area: Área mínima en píxeles para considerar un campo válido
        
        Returns:
            labeled_array: Array con campos etiquetados
            num_fields: Número de campos detectados
        """
        # PASO 1: Expandir barreras para separar mejor los campos
        expanded_barriers = self.expand_barriers(iterations=expand_barriers_iterations)
        
        # PASO 2: Limpiar campos - quitar áreas que están en barreras expandidas
        clean_fields = self.field_mask & ~expanded_barriers
        
        # PASO 3: Aplicar operaciones morfológicas para limpiar ruido
        # Closing: elimina pequeños agujeros
        kernel_close = np.ones((5, 5), dtype=np.uint8)
        clean_fields = ndimage.binary_closing(clean_fields, structure=kernel_close)
        
        # Opening: elimina pequeños puntos aislados
        kernel_open = np.ones((3, 3), dtype=np.uint8)
        clean_fields = ndimage.binary_opening(clean_fields, structure=kernel_open)
        
        # PASO 4: Etiquetar regiones conectadas (8-conectividad)
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_fields = ndimage.label(clean_fields, structure=structure)
        
        # PASO 5: Filtrar campos muy pequeños (probablemente ruido)
        if min_area > 0:
            labeled_array, num_fields = self._filter_small_fields(
                labeled_array, num_fields, min_area
            )
        
        return labeled_array, num_fields
    
    def _filter_small_fields(
        self, 
        labeled_array: np.ndarray, 
        num_fields: int, 
        min_area: int
    ) -> Tuple[np.ndarray, int]:
        """
        Filtra campos que son demasiado pequeños.
        
        Args:
            labeled_array: Array etiquetado
            num_fields: Número de campos
            min_area: Área mínima
            
        Returns:
            labeled_array filtrado y nuevo número de campos
        """
        new_labeled = np.zeros_like(labeled_array)
        new_id = 1
        
        for field_id in range(1, num_fields + 1):
            field_pixels = (labeled_array == field_id)
            area = np.sum(field_pixels)
            
            if area >= min_area:
                new_labeled[field_pixels] = new_id
                new_id += 1
        
        return new_labeled, new_id - 1
    
    def count_meeples_in_field(
        self, 
        field_label: int,
        labeled_fields: np.ndarray,
        meeple_masks: Dict[str, np.ndarray]
    ) -> Dict[str, int]:
        """
        Cuenta meeples de cada jugador en un campo.
        CORREGIDO: Detecta meeples dentro o muy cerca del campo.
        
        Args:
            field_label: ID del campo
            labeled_fields: Array de campos etiquetados
            meeple_masks: Diccionario con máscaras de meeples por jugador
            
        Returns:
            Diccionario con conteo de meeples por jugador
        """
        field_pixels = (labeled_fields == field_label)
        
        # Expandir ligeramente el campo para capturar meeples en el borde
        kernel = np.ones((5, 5), dtype=np.uint8)
        expanded_field = ndimage.binary_dilation(field_pixels, structure=kernel, iterations=2)
        
        counts = {}
        
        for meeple_type, mask in meeple_masks.items():
            # Contar píxeles de meeple dentro del campo expandido
            meeples_in_field = expanded_field & mask
            pixel_count = np.sum(meeples_in_field)
            
            # Contar como meeple si hay suficiente área (>= 10 píxeles)
            # Esto evita ruido pero detecta meeples reales
            if pixel_count >= 10:
                # Contar cuántos meeples individuales hay
                # (en caso de que haya varios meeples del mismo jugador)
                labeled_meeples, num_meeples = ndimage.label(
                    meeples_in_field,
                    structure=np.ones((3, 3), dtype=int)
                )
                counts[meeple_type] = num_meeples
            else:
                counts[meeple_type] = 0
        
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