import os
from datetime import datetime
from typing import Dict
from origin_matrix import Board
from modules.imagen_generator.board_image_generator import BoardImageGenerator
from modules.incomplete_features_scorer.incomplete_features_scorer import GameScorer
from modules.CarcassoneFieldsv5.puntos_campos import calculate_field_scores


##volar
from modules.random_board_generator import generate_board


def run_game():

    ##Metodo de agu y bonatto
    matrix_tiles = generate_board(12) ####
    board_game = Board(board=matrix_tiles) ####
    ##Metodo de agu y bonatto

    ##Genero imagen del tablero
    tiles_img_path = os.path.join("resources","tiles_texture_pack-v3")
    gen_images = BoardImageGenerator("abs/tiles_texture_pack-v3")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join("modules","imagen_generator","output", f"tablero_{timestamp_str}.jpg")
    gen_images.generate_board_image(board_game,image_path)

    ##Calculo puntos
    scores_incomplete_features_scorer = GameScorer(board_game).score()
    scores_fields = calculate_field_scores(image_path)
    
    player1 = scores_fields.get(1) + scores_incomplete_features_scorer(1)
    player2 = scores_fields.get(2) + scores_incomplete_features_scorer(2)
    
    print("jugador 1 = " + player1 + "\njugador 2 = " + player2)

    return


if __name__ == "__main__":
    run_game()