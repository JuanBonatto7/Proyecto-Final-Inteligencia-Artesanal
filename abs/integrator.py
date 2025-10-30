from datetime import datetime
import os
from typing import Dict, Any

from origin_matrix import Board, Tile
from random_board_generator import generate_board
from board_image_generator import BoardImageGenerator
from incomplete_features_scorer import GameScorer, set_debug

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
        tiles_folder="abs/tiles",
        output_folder="output",
    )