"""
Cálculo de puntuación de campos.
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
    
    def count_adjacent_castles(self, field: Field) -> int:
        """
        Cuenta castillos adyacentes a un campo.
        
        Args:
            field: Campo a analizar
            
        Returns:
            Número de castillos adyacentes únicos
        """
        # Dilatar el campo para encontrar vecinos
        dilated = ndimage.binary_dilation(field.pixels, iterations=2)
        
        # Castillos que tocan el campo
        adjacent_castles = dilated & self.castle_mask
        
        if not np.any(adjacent_castles):
            return 0
        
        # Contar regiones de castillos únicas
        labeled_castles, num_castles = ndimage.label(adjacent_castles)
        
        return num_castles
    
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
            score = self.calculate_field_score(field)
            
            results[field.id] = {
                'owner': owner,
                'is_tie': is_tie,
                'score': score,
                'meeples': field.meeples.copy(),
                'castles': score // 3,
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