"""
Programa principal para análisis de campos en Carcassonne.
Incluye detección de límites del tablero y castillos incompletos.
"""
import sys
import os
import cv2
import shutil
import numpy as np

# Configurar encoding para Windows
if sys.platform == 'win32':
    import codecs
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from typing import Dict
from src.image_processor import ImageProcessor
from src.board_detector import BoardDetector
from src.castle_analyzer import CastleAnalyzer
from src.field_detector import FieldDetector
from src.scoring import FieldScorer
from src.visualizer import FieldVisualizer
from config.colors import PLAYER_NAMES, FIELD_DETECTION_CONFIG, WHITE_THRESHOLD


def calculate_field_scores(image_path: str) -> Dict[int, int]:
    """
    Calcula puntos de campos para un tablero.
    FUNCIÓN PÚBLICA PARA INTEGRACIÓN CON OTROS PROGRAMAS.
    """
    # Procesar imagen
    processor = ImageProcessor(image_path)
    
    # Detectar límites del tablero
    board_detector = BoardDetector(processor.image)
    board_mask = board_detector.create_board_mask()
    white_areas = board_detector.detect_white_areas()
    
    # Crear máscaras
    field_mask = board_detector.filter_mask_by_board(processor.create_mask('FIELD'))
    castle_mask = board_detector.filter_mask_by_board(processor.create_mask('CASTLE'))
    barrier_mask = board_detector.filter_mask_by_board(processor.get_combined_barrier_mask())
    
    meeple_masks = {
        'MEEPLE_1': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_1')),
        'MEEPLE_2': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_2')),
    }
    
    # Analizar castillos
    castle_analyzer = CastleAnalyzer(castle_mask, board_detector)
    
    # Detectar campos
    detector = FieldDetector(field_mask, barrier_mask, castle_mask)
    config = FIELD_DETECTION_CONFIG
    labeled_fields, num_fields = detector.detect_fields(
        expand_barriers_iterations=config['barrier_expansion'],
        min_area=config['min_field_area']
    )
    
    road_mask = board_detector.filter_mask_by_board(processor.create_mask('ROAD'))
    fields = detector.create_fields(labeled_fields, num_fields, meeple_masks, road_mask=road_mask)
    
    # Calcular puntuación
    scorer = FieldScorer(castle_mask, castle_analyzer=castle_analyzer)
    field_results = scorer.calculate_all_scores(fields)
    player_totals = scorer.calculate_player_totals(field_results)
    
    # Convertir formato: {'MEEPLE_1': x, 'MEEPLE_2': y} -> {1: x, 2: y}
    return {
        1: player_totals.get('MEEPLE_1', 0),
        2: player_totals.get('MEEPLE_2', 0)
    }


def print_safe(text):
    """Imprime texto de forma segura en cualquier plataforma."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def main(image_path: str, output_path: str = None):
    """
    Ejecuta el análisis completo de campos con visualización.
    
    Args:
        image_path: Ruta de la imagen del tablero
        output_path: Ruta para guardar resultado (opcional)
    """
    # Procesar imagen
    processor = ImageProcessor(image_path)
    
    # Detectar límites del tablero
    board_detector = BoardDetector(processor.image)
    board_mask = board_detector.create_board_mask()
    white_areas = board_detector.detect_white_areas()
    
    # Crear máscaras
    field_mask = processor.create_mask('FIELD')
    castle_mask = processor.create_mask('CASTLE')
    
    # Filtrar máscaras para solo incluir área del tablero
    field_mask = board_detector.filter_mask_by_board(field_mask)
    castle_mask = board_detector.filter_mask_by_board(castle_mask)
    
    # Analizar castillos
    castle_analyzer = CastleAnalyzer(castle_mask, board_detector)
    
    # Crear máscara de barreras (incluye TODOS los castillos + caminos)
    barrier_mask = processor.get_combined_barrier_mask()
    barrier_mask = board_detector.filter_mask_by_board(barrier_mask)
    
    meeple_masks = {
        'MEEPLE_1': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_1')),
        'MEEPLE_2': board_detector.filter_mask_by_board(processor.create_mask('MEEPLE_2')),
    }
    
    # Detectar campos
    detector = FieldDetector(field_mask, barrier_mask, castle_mask)
    
    config = FIELD_DETECTION_CONFIG
    labeled_fields, num_fields = detector.detect_fields(
        expand_barriers_iterations=config['barrier_expansion'],
        min_area=config['min_field_area']
    )
    
    road_mask = processor.create_mask('ROAD')
    road_mask = board_detector.filter_mask_by_board(road_mask)
    fields = detector.create_fields(labeled_fields, num_fields, meeple_masks, road_mask=road_mask)
    
    # Calcular puntuación
    scorer = FieldScorer(castle_mask, castle_analyzer=castle_analyzer)
    field_results = scorer.calculate_all_scores(fields)
    player_totals = scorer.calculate_player_totals(field_results)
    
    print(f"Jugador 1: {player_totals.get('MEEPLE_1', 0)} puntos | Jugador 2: {player_totals.get('MEEPLE_2', 0)} puntos")
    
    return field_results, player_totals