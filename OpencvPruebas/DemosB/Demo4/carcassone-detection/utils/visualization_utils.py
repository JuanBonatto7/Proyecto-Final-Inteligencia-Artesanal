"""
Utilidades para visualización de resultados.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from models.tile import Tile
from models.board import Board


def draw_lines(image: np.ndarray, 
               lines: List[Tuple[int, int, int, int]], 
               color: Tuple[int, int, int] = (0, 255, 0),
               thickness: int = 2) -> np.ndarray:
    """
    Dibuja líneas sobre una imagen.
    
    Args:
        image: Imagen base
        lines: Lista de líneas (x1, y1, x2, y2)
        color: Color BGR
        thickness: Grosor de línea
        
    Returns:
        Imagen con líneas dibujadas
    """
    result = image.copy()
    
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for line in lines:
        x1, y1, x2, y2 = map(int, line)
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    
    return result


def draw_points(image: np.ndarray,
                points: List[Tuple[float, float]],
                color: Tuple[int, int, int] = (0, 0, 255),
                radius: int = 5,
                thickness: int = -1) -> np.ndarray:
    """
    Dibuja puntos sobre una imagen.
    
    Args:
        image: Imagen base
        points: Lista de puntos (x, y)
        color: Color BGR
        radius: Radio del círculo
        thickness: Grosor (-1 = relleno)
        
    Returns:
        Imagen con puntos dibujados
    """
    result = image.copy()
    
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for point in points:
        x, y = map(int, point)
        cv2.circle(result, (x, y), radius, color, thickness)
    
    return result


def draw_rectangles(image: np.ndarray,
                    rectangles: List[np.ndarray],
                    color: Tuple[int, int, int] = (255, 0, 0),
                    thickness: int = 2) -> np.ndarray:
    """
    Dibuja rectángulos sobre una imagen.
    
    Args:
        image: Imagen base
        rectangles: Lista de arrays con 4 esquinas
        color: Color BGR
        thickness: Grosor
        
    Returns:
        Imagen con rectángulos dibujados
    """
    result = image.copy()
    
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for rect in rectangles:
        points = rect.astype(np.int32)
        cv2.polylines(result, [points], True, color, thickness)
    
    return result


def draw_tiles(image: np.ndarray, tiles: List[Tile]) -> np.ndarray:
    """
    Dibuja las fichas detectadas con sus etiquetas.
    
    Args:
        image: Imagen base
        tiles: Lista de fichas detectadas
        
    Returns:
        Imagen con fichas dibujadas
    """
    result = image.copy()
    
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for tile in tiles:
        if tile.corners is not None:
            # Dibujar contorno
            points = tile.corners.astype(np.int32)
            cv2.polylines(result, [points], True, (0, 255, 0), 2)
            
            # Calcular centro
            center = tuple(np.mean(points, axis=0).astype(int))
            
            # Dibujar etiqueta
            label = f"{tile.tile_type}_{tile.rotation}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Fondo para el texto
            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            cv2.rectangle(
                result,
                (center[0] - text_width // 2 - 5, center[1] - text_height // 2 - 5),
                (center[0] + text_width // 2 + 5, center[1] + text_height // 2 + 5),
                (255, 255, 255),
                -1
            )
            
            # Texto
            cv2.putText(
                result,
                label,
                (center[0] - text_width // 2, center[1] + text_height // 2),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness
            )
    
    return result


def create_board_visualization(board: Board) -> np.ndarray:
    """
    Crea visualización de la matriz del tablero.
    
    Args:
        board: Tablero a visualizar
        
    Returns:
        Imagen con representación del tablero
    """
    cell_size = 80
    padding = 5
    
    height = board.rows * cell_size
    width = board.cols * cell_size
    
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    for row in range(board.rows):
        for col in range(board.cols):
            # Posición de la celda
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            
            # Dibujar borde
            cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), 1)
            
            # Obtener ficha
            tile = board.get_tile_at(row, col)
            
            if tile is not None:
                # Fondo de celda ocupada
                cv2.rectangle(image, (x1, y1), (x2, y2), (230, 255, 230), -1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), 1)
                
                # Texto
                label = f"{tile.tile_type}"
                rotation_label = f"R{tile.rotation}"
                
                # Centrar texto
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_x = x1 + (cell_size - tw) // 2
                text_y = y1 + (cell_size + th) // 2 - 10
                
                cv2.putText(
                    image, label, (text_x, text_y),
                    font, font_scale, (0, 0, 0), font_thickness
                )
                
                # Rotación
                (tw2, th2), _ = cv2.getTextSize(rotation_label, font, font_scale - 0.1, font_thickness)
                text_x2 = x1 + (cell_size - tw2) // 2
                text_y2 = text_y + th + 5
                
                cv2.putText(
                    image, rotation_label, (text_x2, text_y2),
                    font, font_scale - 0.1, (100, 100, 100), font_thickness
                )
    
    return image