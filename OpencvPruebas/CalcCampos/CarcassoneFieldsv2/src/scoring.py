"""
Cálculo de puntuación de campos (CORREGIDO).
"""
from typing import Dict, List, Tuple
from src.field_detector import Field
import numpy as np
from scipy import ndimage


class FieldScorer:
    """Calcula puntuación de campos."""
    
    def __init__(self, castle_mask: np.ndarray):
        """
        Inicializa el calculador de puntos.
        
        Args:
            castle_mask: Máscara de castillos
        """
        self.castle_mask = castle_mask
        # Etiquetar castillos individuales al inicio
        self.labeled_castles, self.num_castles = ndimage.label(
            castle_mask, 
            structure=np.ones((3, 3), dtype=int)
        )
    
    def get_field_boundary(self, field: Field) -> np.ndarray:
        """
        Obtiene los píxeles del borde del campo.
        
        Args:
            field: Campo a analizar
            
        Returns:
            Máscara binaria con solo el borde del campo
        """
        # Erosión para obtener el interior
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = ndimage.binary_erosion(field.pixels, structure=kernel)
        
        # Borde = campo original - interior erosionado
        boundary = field.pixels & ~eroded
        
        return boundary
    
    def count_adjacent_castles(self, field: Field) -> int:
        """
        Cuenta castillos adyacentes a un campo.
        CORREGIDO: Detecta castillos que tocan el borde del campo.
        
        Args:
            field: Campo a analizar
            
        Returns:
            Número de castillos adyacentes únicos
        """
        # Obtener el borde del campo
        boundary = self.get_field_boundary(field)
        
        # Expandir el borde ligeramente para detectar adyacencia
        kernel = np.ones((5, 5), dtype=np.uint8)
        expanded_boundary = ndimage.binary_dilation(boundary, structure=kernel, iterations=2)
        
        # Encontrar qué castillos intersectan con el borde expandido
        adjacent_castle_region = expanded_boundary & self.castle_mask
        
        if not np.any(adjacent_castle_region):
            return 0
        
        # Identificar qué castillos únicos están tocando
        unique_castle_ids = set()
        
        # Para cada píxel del borde expandido que toca un castillo
        y_coords, x_coords = np.where(adjacent_castle_region)
        
        for y, x in zip(y_coords, x_coords):
            castle_id = self.labeled_castles[y, x]
            if castle_id > 0:  # 0 es el fondo
                unique_castle_ids.add(castle_id)
        
        return len(unique_castle_ids)
    
    def determine_owner(self, field: Field) -> Tuple[str, bool]:
        """
        Determina el dueño de un campo.
        
        Args:
            field: Campo a analizar
            
        Returns:
            (owner, is_tie): Tupla con el dueño y si hay empate
        """
        if not field.meeples or all(count == 0 for count in field.meeples.values()):
            return None, False
        
        max_count = max(field.meeples.values())
        owners = [player for player, count in field.meeples.items() if count == max_count]
        
        if len(owners) > 1:
            return 'TIE', True
        
        return owners[0], False
    
    def calculate_field_score(self, field: Field) -> int:
        """
        Calcula puntos de un campo (3 puntos por castillo adyacente).
        
        Args:
            field: Campo a puntuar
            
        Returns:
            Puntos del campo
        """
        num_castles = self.count_adjacent_castles(field)
        return num_castles * 3
    
    def calculate_all_scores(
        self, 
        fields: List[Field]
    ) -> Dict[str, Dict]:
        """
        Calcula puntuación para todos los campos.
        
        Args:
            fields: Lista de campos
            
        Returns:
            Diccionario con información de puntuación por campo
        """
        results = {}
        
        for field in fields:
            owner, is_tie = self.determine_owner(field)
            castles = self.count_adjacent_castles(field)
            score = castles * 3
            
            results[field.id] = {
                'owner': owner,
                'is_tie': is_tie,
                'score': score,
                'meeples': field.meeples.copy(),
                'castles': castles,
                'area': field.area
            }
        
        return results
    
    def calculate_player_totals(
        self, 
        field_results: Dict[str, Dict]
    ) -> Dict[str, int]:
        """
        Calcula puntos totales por jugador.
        
        Args:
            field_results: Resultados por campo
            
        Returns:
            Puntos totales por jugador
        """
        totals = {}
        
        for field_data in field_results.values():
            owner = field_data['owner']
            score = field_data['score']
            is_tie = field_data['is_tie']
            
            if owner and owner != 'TIE':
                totals[owner] = totals.get(owner, 0) + score
            elif is_tie:
                # En caso de empate, todos los jugadores empatados reciben puntos
                for player, count in field_data['meeples'].items():
                    if count == max(field_data['meeples'].values()) and count > 0:
                        totals[player] = totals.get(player, 0) + score
        
        return totals