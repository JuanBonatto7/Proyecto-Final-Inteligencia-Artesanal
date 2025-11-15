# ============================================================================
# scorers/field_scorer.py
# ============================================================================

from CalcCampos.carcassonne_scorer.models.game_state import GameState
from CalcCampos.carcassonne_scorer.processors.castle_detector import CastleDetector
from CalcCampos.carcassonne_scorer.processors.field_detector import FieldDetector
from CalcCampos.carcassonne_scorer.processors.image_processor import ImageProcessor


class FieldScorer:
    """Calculador de puntuación de campos."""
    
    def __init__(self, 
                 image_processor: ImageProcessor,
                 field_detector: FieldDetector,
                 castle_detector: CastleDetector):
        self.processor = image_processor
        self.field_detector = field_detector
        self.castle_detector = castle_detector
    
    def score_fields(self) -> GameState:
        """
        Procesa todos los campos y calcula puntuaciones.
        
        Returns:
            Estado del juego con puntuaciones
        """
        game_state = GameState()
        
        # Detectar campos
        fields = self.field_detector.detect_fields()
        
        print(f"\n{'='*60}")
        print(f"CAMPOS DETECTADOS: {len(fields)}")
        print(f"{'='*60}\n")
        
        # Procesar cada campo
        for field in fields:
            # Contar meeples
            self.field_detector.count_meeples_in_field(field)
            
            # Contar castillos CERRADOS que tocan este campo
            field.closed_castles_touching = self.castle_detector.count_closed_castles_touching_field(field)
            
            # Añadir al estado del juego
            game_state.add_field(field)
            
            # Imprimir información
            print(field)
        
        # Calcular puntuaciones totales
        scores = game_state.calculate_scores()
        
        print(f"\n{'='*60}")
        print(f"PUNTUACIÓN TOTAL")
        print(f"{'='*60}")
        print(f"Jugador 1: {scores[1]} puntos")
        print(f"Jugador 2: {scores[2]} puntos")
        print(f"{'='*60}\n")
        
        return game_state
