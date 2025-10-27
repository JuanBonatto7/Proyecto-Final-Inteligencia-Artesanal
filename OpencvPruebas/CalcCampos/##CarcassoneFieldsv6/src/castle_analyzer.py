"""
Análisis de castillos completos vs incompletos.
Un castillo incompleto toca el borde del tablero (área blanca).
"""
import numpy as np
from scipy import ndimage
from typing import Set, Dict


class CastleAnalyzer:
    """Analiza castillos y determina cuáles están completos."""
    
    def __init__(self, castle_mask: np.ndarray, board_detector):
        """
        Inicializa el analizador de castillos.
        
        Args:
            castle_mask: Máscara binaria de todos los castillos
            board_detector: Instancia de BoardDetector
        """
        self.castle_mask = castle_mask
        self.board_detector = board_detector
        
        # Etiquetar cada castillo individualmente
        self.labeled_castles, self.num_castles = ndimage.label(
            castle_mask,
            structure=np.ones((3, 3), dtype=int)
        )
        
        # Analizar cuáles están completos
        self.complete_castles = self._identify_complete_castles()
        self.incomplete_castles = self._identify_incomplete_castles()
    
    def _identify_complete_castles(self) -> Set[int]:
        """
        Identifica castillos completos.
        Un castillo está incompleto si tiene blanco penetrando en su estructura.

        Returns:
            Set de IDs de castillos completos
        """
        complete = set()
        white_areas = self.board_detector.detect_white_areas()
    
        for castle_id in range(1, self.num_castles + 1):
            castle_pixels = (self.labeled_castles == castle_id)

            # Dilatar el castillo moderadamente
            kernel = np.ones((5, 5), dtype=np.uint8)
            dilated_castle = ndimage.binary_dilation(castle_pixels, structure=kernel, iterations=2)

            # Encontrar "cavidades" - áreas blancas dentro del castillo dilatado
            white_inside = dilated_castle & white_areas

            # Si hay blanco significativo dentro del castillo dilatado = incompleto
            white_pixel_count = np.sum(white_inside)

            # Umbral: si hay más de 50 píxeles blancos dentro = incompleto
            # (ajustar según tamaño de losetas)
            if white_pixel_count < 50:
                complete.add(castle_id)
    
        return complete
    
    def _identify_incomplete_castles(self) -> Set[int]:
        """
        Identifica castillos incompletos (tocan el blanco).
        
        Returns:
            Set de IDs de castillos incompletos
        """
        incomplete = set()
        
        for castle_id in range(1, self.num_castles + 1):
            if castle_id not in self.complete_castles:
                incomplete.add(castle_id)
        
        return incomplete
    
    def is_castle_complete(self, castle_id: int) -> bool:
        """
        Verifica si un castillo específico está completo.
        
        Args:
            castle_id: ID del castillo (1 a num_castles)
            
        Returns:
            True si el castillo está completo
        """
        return castle_id in self.complete_castles
    
    def get_complete_castles_mask(self) -> np.ndarray:
        """
        Obtiene máscara de solo castillos completos.
        
        Returns:
            Máscara binaria con solo castillos completos
        """
        complete_mask = np.zeros_like(self.castle_mask, dtype=bool)
        
        for castle_id in self.complete_castles:
            complete_mask |= (self.labeled_castles == castle_id)
        
        return complete_mask
    
    def get_incomplete_castles_mask(self) -> np.ndarray:
        """
        Obtiene máscara de solo castillos incompletos.
        
        Returns:
            Máscara binaria con solo castillos incompletos
        """
        incomplete_mask = np.zeros_like(self.castle_mask, dtype=bool)
        
        for castle_id in self.incomplete_castles:
            incomplete_mask |= (self.labeled_castles == castle_id)
        
        return incomplete_mask
    
    def get_castle_statistics(self) -> Dict:
        """
        Obtiene estadísticas de castillos.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'total_castles': self.num_castles,
            'complete_castles': len(self.complete_castles),
            'incomplete_castles': len(self.incomplete_castles),
            'complete_ids': list(self.complete_castles),
            'incomplete_ids': list(self.incomplete_castles)
        }