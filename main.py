import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from origin_matrix import Board
from modules.piplineFotoRecorteMeepleTipo.origin_matrix_converter import OriginMatrixConverter
from modules.imagen_generator.board_image_generator import BoardImageGenerator
from modules.incomplete_features_scorer.incomplete_features_scorer import GameScorer
from modules.CarcassoneFieldsv5.puntos_campos import calculate_field_scores


##volar
from modules.random_board_generator import generate_board


def process_board_image(
    image_path: str,
    timestamp: Optional[str] = None,
    tiles_texture_pack: str = "tiles_texture_pack-v4",
    web_mode: bool = False,
    reference_coords: Optional[list] = None,
    manual_selections: Optional[dict] = None
) -> Tuple[Dict[int, int], str, object]:
    """
    Procesa una imagen de tablero de Carcassonne y calcula los puntajes.
    
    Args:
        image_path: Ruta a la imagen del tablero a procesar
        timestamp: Timestamp para nombrar archivos generados (si None, se genera uno nuevo)
        tiles_texture_pack: Nombre del pack de texturas a usar (default: tiles_texture_pack-v4)
        web_mode: Si True, procesa en modo web (para confirmaciones interactivas)
        reference_coords: Coordenadas de las losetas de referencia (modo web)
        manual_selections: Selecciones manuales de losetas (modo web)
    
    Returns:
        Tupla con (scores_dict, output_image_path, board_game_object)
        scores_dict contiene: {
            1: puntaje_jugador1,
            2: puntaje_jugador2,
            'player1_fields': puntaje_campos_jugador1,
            'player1_features': puntaje_características_jugador1,
            'player2_fields': puntaje_campos_jugador2,
            'player2_features': puntaje_características_jugador2
        }
    """
    # Convertir imagen a board
    origin_convert = OriginMatrixConverter()
    board_game = origin_convert.convert(
        image_path,
        web_mode=web_mode,
        reference_coords=reference_coords,
        manual_selections=manual_selections
    )
    
    # Si devuelve un dict con needs_confirmation, retornarlo tal cual
    if isinstance(board_game, dict) and board_game.get('needs_confirmation', False):
        return board_game, None, None
    
    # Generar timestamp si no se proporciona
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generar imagen del tablero
    tiles_img_path = os.path.join("resources", tiles_texture_pack)
    gen_images = BoardImageGenerator(tiles_img_path)
    output_image_path = os.path.join("modules", "imagen_generator", "output", f"tablero_{timestamp}.jpg")
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    gen_images.generate_board_image(board_game, output_image_path)
    
    # Calcular puntos
    scores_incomplete_features_scorer = GameScorer(board_game).score()
    scores_fields = calculate_field_scores(output_image_path)
    
    player1_fields = scores_fields.get(1, 0)
    player1_features = scores_incomplete_features_scorer.get(1, 0)
    player2_fields = scores_fields.get(2, 0)
    player2_features = scores_incomplete_features_scorer.get(2, 0)
    
    player1_total = player1_fields + player1_features
    player2_total = player2_fields + player2_features
    
    scores = {
        1: player1_total,
        2: player2_total,
        'player1_fields': player1_fields,
        'player1_features': player1_features,
        'player2_fields': player2_fields,
        'player2_features': player2_features
    }
    
    return scores, output_image_path, board_game


def run_game():

    ##Metodo para generar board aleatorios
    ##matrix_tiles = generate_board(12)      ####
    ##board_game = Board(board=matrix_tiles) ####
    ##Metodo para generar board aleatorios

    scores, image_path, board_game = process_board_image("tablero.jpg")
    
    print("jugador 1 = " + str(scores[1]) + "\njugador 2 = " + str(scores[2]))

    return


if __name__ == "__main__":
    run_game()