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
    
    def __init__(self, field_mask: np.ndarray, barrier_mask: np.ndarray, castle_mask: np.ndarray = None):
        """
        Inicializa el detector de campos.
        
        Args:
            field_mask: Máscara de campos (verde)
            barrier_mask: Máscara de barreras (caminos + castillos)
            castle_mask: Máscara de castillos (opcional, para validar meeples)
        """
        self.field_mask = field_mask
        self.barrier_mask = barrier_mask
        self.castle_mask = castle_mask
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
        meeple_masks: Dict[str, np.ndarray],
        road_mask: np.ndarray = None  # NUEVO parámetro
    ) -> Dict[str, int]:
        """
        Cuenta meeples de cada jugador en un campo.
        Excluye meeples que tocan caminos (están en caminos, no en campos).
        Excluye meeples que están completamente dentro de castillos (deben tocar campo).
        """
        field_pixels = (labeled_fields == field_label)

        # Expandir campo
        kernel = np.ones((5, 5), dtype=np.uint8)
        expanded_field = ndimage.binary_dilation(field_pixels, structure=kernel, iterations=2)

        counts = {}

        for meeple_type, mask in meeple_masks.items():
            # Meeples en el campo
            meeples_in_field = expanded_field & mask
            pixel_count = np.sum(meeples_in_field)

            if pixel_count >= 10:
                # NUEVO: Verificar si el meeple toca caminos
                if road_mask is not None:
                    # Expandir meeple ligeramente para detectar proximidad
                    expanded_meeple = ndimage.binary_dilation(meeples_in_field, iterations=2)
                    touches_road = np.any(expanded_meeple & road_mask)

                    # Si toca camino, NO cuenta para campo
                    if touches_road:
                        counts[meeple_type] = 0
                        continue
                
                # NUEVO: Verificar si el meeple está completamente dentro de un castillo
                # Si está completamente dentro de un castillo, debe ser descartado
                if self.castle_mask is not None:
                    # Obtener píxeles originales del meeple (sin expandir)
                    original_meeple = mask & (labeled_fields == field_label)
                    if np.sum(original_meeple) > 0:
                        # Expandir ligeramente el meeple para detectar contacto
                        expanded_meeple_for_field = ndimage.binary_dilation(original_meeple, iterations=2)
                        
                        # Verificar si toca algún píxel verde (campo real)
                        touches_actual_field = np.any(expanded_meeple_for_field & self.field_mask)
                        
                        # Si NO toca ningún campo verde, está completamente en castillo
                        if not touches_actual_field:
                            counts[meeple_type] = 0
                            continue
                
                # MEJORADO: Dilatar antes de etiquetar para agrupar fragmentos del mismo meeple
                # Esto evita contar un meeple fragmentado como múltiples meeples
                dilated_meeples = ndimage.binary_dilation(
                    meeples_in_field, 
                    structure=np.ones((5, 5), dtype=int),
                    iterations=3
                )
                    
                # Contar meeples individuales
                labeled_meeples, num_meeples = ndimage.label(
                    dilated_meeples,
                    structure=np.ones((3, 3), dtype=int)
                )
                counts[meeple_type] = num_meeples
            else:
                counts[meeple_type] = 0

        return counts
    
    def analyze_meeple_validity(
        self,
        meeple_masks: Dict[str, np.ndarray],
        labeled_fields: np.ndarray,
        road_mask: np.ndarray = None
    ) -> Dict[str, Dict[str, List[np.ndarray]]]:
        """
        Analiza todos los meeples y clasifica cuáles son válidos e inválidos.
        
        Args:
            meeple_masks: Máscaras de meeples por jugador
            labeled_fields: Campos etiquetados
            road_mask: Máscara de caminos
            
        Returns:
            Diccionario con 'valid' e 'invalid' para cada tipo de meeple,
            conteniendo listas de máscaras de meeples individuales
        """
        result = {}
        
        for meeple_type, mask in meeple_masks.items():
            valid_meeples = []
            invalid_meeples = []
            
            # Etiquetar meeples individuales (con dilatación para agrupar fragmentos)
            dilated_mask = ndimage.binary_dilation(
                mask,
                structure=np.ones((5, 5), dtype=int),
                iterations=3
            )
            labeled_meeples, num_meeples = ndimage.label(
                dilated_mask,
                structure=np.ones((3, 3), dtype=int)
            )
            
            # Analizar cada meeple
            for meeple_id in range(1, num_meeples + 1):
                meeple_pixels = (labeled_meeples == meeple_id)
                
                # Verificar si tiene suficientes píxeles
                if np.sum(meeple_pixels) < 10:
                    continue
                
                # Verificar si toca algún campo
                expanded_meeple = ndimage.binary_dilation(
                    meeple_pixels,
                    structure=np.ones((5, 5), dtype=int),
                    iterations=2
                )
                touches_field = np.any(expanded_meeple & (labeled_fields > 0))
                
                # Verificar si toca caminos
                touches_road = False
                if road_mask is not None:
                    expanded_for_road = ndimage.binary_dilation(meeple_pixels, iterations=2)
                    touches_road = np.any(expanded_for_road & road_mask)
                
                # NUEVO: Verificar si está completamente dentro de un castillo
                inside_castle = False
                if self.castle_mask is not None:
                    expanded_for_field = ndimage.binary_dilation(meeple_pixels, iterations=2)
                    touches_actual_field = np.any(expanded_for_field & self.field_mask)
                    # Si no toca campo verde, está completamente en castillo
                    inside_castle = not touches_actual_field
                
                # Clasificar meeple
                is_valid = touches_field and not touches_road and not inside_castle
                
                if is_valid:
                    valid_meeples.append(meeple_pixels)
                else:
                    invalid_meeples.append(meeple_pixels)
            
            result[meeple_type] = {
                'valid': valid_meeples,
                'invalid': invalid_meeples
            }
        
        return result
    
    def create_fields(
        self, 
        labeled_fields: np.ndarray,
        num_fields: int,
        meeple_masks: Dict[str, np.ndarray],
        road_mask: np.ndarray = None  # NUEVO parámetro
    ) -> List[Field]:
        """Crea objetos Field para cada campo detectado."""
        fields = []
        
        for field_id in range(1, num_fields + 1):
            field_pixels = (labeled_fields == field_id)
            area = np.sum(field_pixels)
            
            if area == 0:
                continue
            
            # PASAR road_mask al contador
            meeples = self.count_meeples_in_field(
                field_id, 
                labeled_fields, 
                meeple_masks,
                road_mask=road_mask  # NUEVO
            )
            
            field = Field(
                id=field_id,
                pixels=field_pixels,
                meeples=meeples,
                area=area
            )
            
            fields.append(field)
        
        return fields