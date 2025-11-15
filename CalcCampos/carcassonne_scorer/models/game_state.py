# ============================================================================
# models/game_state.py
# ============================================================================

from dataclasses import Field
from typing import List, Dict

class GameState:
    """Estado del juego con todos los campos detectados."""
    
    def __init__(self):
        self.fields: List[Field] = []
        self.total_scores: Dict[int, int] = {1: 0, 2: 0}
    
    def add_field(self, field: Field):
        """Añade un campo al estado del juego."""
        self.fields.append(field)
    
    def calculate_scores(self) -> Dict[int, int]:
        """Calcula los puntos totales de cada jugador."""
        scores = {1: 0, 2: 0}
        
        for field in self.fields:
            if not field.is_valid:
                continue
                
            owner = field.owner
            points = field.points
            
            if owner == 1:
                scores[1] += points
            elif owner == 2:
                scores[2] += points
            elif owner == 0:  # Empate
                scores[1] += points
                scores[2] += points
        
        self.total_scores = scores
        return scores
    
    def get_field_scores(self) -> List[Dict]:
        """Retorna información de puntuación por campo."""
        field_scores = []
        
        for field in self.fields:
            if not field.is_valid or field.owner is None:
                continue
            
            field_info = {
                'field_id': field.id,
                'owner': field.owner,
                'points': field.points,
                'closed_castles': field.closed_castles_touching,
                'meeples': (field.meeples_p1, field.meeples_p2)
            }
            field_scores.append(field_info)
        
        return field_scores