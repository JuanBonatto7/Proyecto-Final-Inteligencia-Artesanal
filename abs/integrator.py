from datetime import datetime
import os
from typing import Dict, Any

from origin_matrix import Board, Tile
from random_board_generator import generate_board
from board_image_generator import BoardImageGenerator
from incomplete_features_scorer import GameScorer, set_debug
from carcassone_field_scorer_v5 import fields_scorer

def _add_field_points_from_image(scores: dict, image_path: str, white_threshold: int = 200) -> dict:
    """Helper mínimo: intenta importar y ejecutar `puntos_campos.main` en modo headless.
    - Suma `MEEPLE_1` a `scores[1]` y `MEEPLE_2` a `scores[2]` cuando están disponibles.
    """
    # Usar el módulo `fields_scorer` importado al inicio
    if 'fields_scorer' not in globals() or fields_scorer is None:
        return scores

    try:
        # Llamada al main del scorer de campos (puede abrir ventanas si el módulo las muestra)
        field_results, player_totals = fields_scorer.main(image_path, output_path=None, white_threshold=white_threshold)
        return _sum_field_points(scores, player_totals)
    except Exception:
        return scores


def _sum_field_points(scores: dict, player_totals: dict) -> dict:
    if not player_totals:
        return scores
    scores[1] = scores.get(1, 0) + int(player_totals.get('MEEPLE_1', 0))
    scores[2] = scores.get(2, 0) + int(player_totals.get('MEEPLE_2', 0))
    return scores


def run_game(
    board_size: int = 12,
    tile_size_px: int = 200,
    tiles_folder: str = "tiles",
    output_folder: str = "output",
    seed: int = 123456,
) -> Dict[str, Any]:
    # Generar tablero (sin prints)
    matrix_tiles = generate_board(board_size)
    board = Board(board=matrix_tiles)

    tiles_count = sum(1 for row in matrix_tiles for tile in row if tile is not None)

    # Generar imagen (silencioso)
    os.makedirs(output_folder, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(output_folder, f"tablero_{timestamp_str}.jpg")
    img_gen = BoardImageGenerator(tiles_folder=tiles_folder, tile_size=tile_size_px)
    _ = img_gen.generate_board_image(board, output_path=image_path, verbose=False)

    # Calcular puntuación (los logs los maneja el scorer vía set_debug)
    scores = GameScorer(board).score()

    # Intentar sumar puntos de campos (helper mínimo)
    #image_path es la imagen generada del tablero
    scores = _add_field_points_from_image(scores, image_path, white_threshold=200)

    # Mostrar solo resultados finales
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    p1 = scores.get(1, 0)
    p2 = scores.get(2, 0)
    print(f"Jugador 1: {p1} puntos")
    print(f"Jugador 2: {p2} puntos")

    return {
        "board": board,
        "tiles_count": tiles_count,
        "image_path": image_path,
        "scores": scores,
        "timestamp": timestamp_str,
    }

if __name__ == "__main__":
    # Mostrar solo los logs de meeples que suman puntos
    set_debug(True, show_positions=False)
    run_game(
        board_size=12,
        tile_size_px=200,
        tiles_folder="tiles_texture_pack-v2",
        output_folder="output",
    )