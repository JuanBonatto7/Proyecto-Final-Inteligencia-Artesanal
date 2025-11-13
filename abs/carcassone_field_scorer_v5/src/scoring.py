"""Módulo de puntuación de campos.

Reglas:
- Sólo los castillos completos aportan puntos.
- Los castillos incompletos delimitan pero no puntúan.

El archivo mantiene la lógica existente; aquí solo se limpian
los comentarios y se usan nombres en formato apropiado.
"""
from typing import Dict, List, Tuple
from .field_detector import Field
import numpy as np
from scipy import ndimage


class FieldScorer:
    """Calculador de puntuación de campos."""

    def __init__(self, castle_mask: np.ndarray, castle_analyzer=None):
        """Inicializa con la máscara de castillos y un analizador opcional."""
        self.castle_mask = castle_mask
        self.castle_analyzer = castle_analyzer
        self.labeled_castles, self.num_castles = ndimage.label(
            castle_mask, structure=np.ones((3, 3), dtype=int)
        )
        if castle_analyzer:
            complete_mask = castle_analyzer.get_complete_castles_mask()
            self.labeled_complete_castles, self.num_complete_castles = ndimage.label(
                complete_mask, structure=np.ones((3, 3), dtype=int)
            )
        else:
            self.labeled_complete_castles = self.labeled_castles
            self.num_complete_castles = self.num_castles

    def count_adjacent_castles(self, field: Field, only_complete: bool = True) -> int:
        """Cuenta castillos únicos adyacentes o dentro de un campo.

        only_complete: si True considera sólo castillos completos.
        Devuelve el número de castillos únicos encontrados.
        """
        kernel = np.ones((7, 7), dtype=np.uint8)
        expanded_field = ndimage.binary_dilation(field.pixels, structure=kernel, iterations=3)
        if only_complete and self.castle_analyzer:
            labeled_to_use = self.labeled_complete_castles
            castles_in_or_near_field = expanded_field & self.castle_analyzer.get_complete_castles_mask()
        else:
            labeled_to_use = self.labeled_castles
            castles_in_or_near_field = expanded_field & self.castle_mask
        if not np.any(castles_in_or_near_field):
            return 0
        unique_castle_ids = set()
        y_coords, x_coords = np.where(castles_in_or_near_field)
        for y, x in zip(y_coords, x_coords):
            castle_id = labeled_to_use[y, x]
            if castle_id > 0:
                unique_castle_ids.add(castle_id)
        return len(unique_castle_ids)

    def determine_owner(self, field: Field) -> Tuple[str, bool]:
        """Determina el dueño del campo y si hay empate.

        Retorna una tupla (owner, is_tie). Si no hay meeples retorna (None, False).
        """
        if not field.meeples or all(count == 0 for count in field.meeples.values()):
            return None, False
        max_count = max(field.meeples.values())
        owners = [player for player, count in field.meeples.items() if count == max_count]
        if len(owners) > 1:
            return 'TIE', True
        return owners[0], False

    def calculate_field_score(self, field: Field) -> int:
        """Devuelve la puntuación del campo (3 puntos por castillo completo)."""
        num_castles = self.count_adjacent_castles(field, only_complete=True)
        return num_castles * 3

    def calculate_all_scores(self, fields: List[Field]) -> Dict[str, Dict]:
        """Calcula y devuelve un diccionario con los resultados de cada campo.

        Cada entrada contiene: owner, is_tie, score, meeples, castles (total),
        castles_complete, castles_incomplete y area.
        """
        results = {}
        for field in fields:
            owner, is_tie = self.determine_owner(field)
            complete_castles = self.count_adjacent_castles(field, only_complete=True)
            score = complete_castles * 3
            total_castles = self.count_adjacent_castles(field, only_complete=False)

            results[field.id] = {
                'owner': owner,
                'is_tie': is_tie,
                'score': score,
                'meeples': field.meeples.copy(),
                'castles': total_castles,
                'castles_complete': complete_castles,
                'castles_incomplete': max(0, total_castles - complete_castles),
                'area': field.area
            }
        return results

    def calculate_player_totals(self, field_results: Dict[str, Dict]) -> Dict[str, int]:
        """Suma los puntos por jugador a partir de los resultados de campos.

        Maneja empates distribuyendo los puntos a los jugadores que comparten
        la mayoría de meeples en el campo.
        """
        # Inicializar totales con las claves esperadas por el proyecto
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
                for player, count in field_data['meeples'].items():
                    if count == max(field_data['meeples'].values()) and count > 0:
                        totals[player] = totals.get(player, 0) + score
        return totals