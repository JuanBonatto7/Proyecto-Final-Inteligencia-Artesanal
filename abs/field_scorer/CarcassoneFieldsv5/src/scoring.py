"""
Cálculo de puntuación de campos.
Solo cuenta castillos COMPLETOS para puntos.
Castillos incompletos delimitan pero no puntúan.
"""
from typing import Dict, List, Tuple
from src.field_detector import Field
import numpy as np
from scipy import ndimage


class FieldScorer:
    """Calcula puntuación de campos."""
    
    def __init__(self, castle_mask: np.ndarray, castle_analyzer=None):
        """
        Inicializa el calculador de puntos.
        
        Args:
            castle_mask: Máscara de TODOS los castillos
            castle_analyzer: Analizador de castillos (opcional)
        """
        self.castle_mask = castle_mask
        self.castle_analyzer = castle_analyzer
        
        # Etiquetar castillos individuales
        self.labeled_castles, self.num_castles = ndimage.label(
            castle_mask, 
            structure=np.ones((3, 3), dtype=int)
        )
        
        # Si tenemos analizador, crear etiquetado de solo castillos completos
        if castle_analyzer:
            complete_mask = castle_analyzer.get_complete_castles_mask()
            self.labeled_complete_castles, self.num_complete_castles = ndimage.label(
                complete_mask,
                structure=np.ones((3, 3), dtype=int)
            )
        else:
            # Fallback: asumir todos completos si no hay analizador
            self.labeled_complete_castles = self.labeled_castles
            self.num_complete_castles = self.num_castles
    
    def count_adjacent_castles(self, field: Field, only_complete: bool = True) -> int:
        """
        Cuenta castillos adyacentes o dentro de un campo.
        
        Args:
            field: Campo a analizar
            only_complete: Si True, solo cuenta castillos completos
            
        Returns:
            Número de castillos adyacentes únicos
        """
        # Expandir el campo para detectar castillos en el borde o dentro
        kernel = np.ones((7, 7), dtype=np.uint8)
        expanded_field = ndimage.binary_dilation(field.pixels, structure=kernel, iterations=3)
        
        # Elegir qué castillos contar
        if only_complete and self.castle_analyzer:
            # Solo castillos COMPLETOS cuentan para puntos
            labeled_to_use = self.labeled_complete_castles
            castles_in_or_near_field = expanded_field & self.castle_analyzer.get_complete_castles_mask()
        else:
            # Todos los castillos (usado para barreras)
            labeled_to_use = self.labeled_castles
            castles_in_or_near_field = expanded_field & self.castle_mask
        
        if not np.any(castles_in_or_near_field):
            return 0
        
        # Identificar qué castillos únicos están presentes
        unique_castle_ids = set()
        
        y_coords, x_coords = np.where(castles_in_or_near_field)
        
        for y, x in zip(y_coords, x_coords):
            castle_id = labeled_to_use[y, x]
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
        Calcula puntos de un campo.
        SOLO cuenta castillos COMPLETOS (3 puntos c/u).
        
        Args:
            field: Campo a puntuar
            
        Returns:
            Puntos del campo
        """
        num_castles = self.count_adjacent_castles(field, only_complete=True)
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
            
            # Castillos COMPLETOS para puntos
            complete_castles = self.count_adjacent_castles(field, only_complete=True)
            score = complete_castles * 3
            
            # Total de castillos (completos + incompletos) para info
            total_castles = self.count_adjacent_castles(field, only_complete=False)
            
            results[field.id] = {
                'owner': owner,
                'is_tie': is_tie,
                'score': score,
                'meeples': field.meeples.copy(),
                'castles': complete_castles,  # Solo completos para puntos
                'castles_complete': complete_castles,
                'castles_incomplete': total_castles - complete_castles,
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
        # Inicializar SIEMPRE ambos jugadores
        totals = {
            'MEEPLE_1': 0,
            'MEEPLE_2': 0,
        }
        
        for field_data in field_results.values():
            owner = field_data['owner']
            score = field_data['score']
            is_tie = field_data['is_tie']
            
            if owner and owner != 'TIE':
                totals[owner] = totals.get(owner, 0) + score
            elif is_tie:
                # En caso de empate, todos reciben puntos
                for player, count in field_data['meeples'].items():
                    if count == max(field_data['meeples'].values()) and count > 0:
                        totals[player] = totals.get(player, 0) + score
        
        return totals