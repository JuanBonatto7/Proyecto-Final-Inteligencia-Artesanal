"""
Detector del tablero y creador de matriz
"""

import cv2
import numpy as np
from config import *
from tile_matcher import TileMatcher
from utils import draw_grid, save_debug_image


class BoardDetector:
    def __init__(self):
        self.matcher = TileMatcher()
        self.image = None
        self.tile_size = TILE_SIZE
        self.reference_position = None
    
    def load_image(self, path):
        """
        Carga la imagen del tablero
        """
        self.image = cv2.imread(path)
        if self.image is None:
            raise Exception(f"No se pudo cargar la imagen: {path}")
        print(f"[INFO] Imagen cargada: {self.image.shape[1]}x{self.image.shape[0]}")
    
    def detect_reference_square(self):
        """
        Detecta el cuadrado de referencia
        ASUMIMOS que está centrado en la imagen
        """
        h, w = self.image.shape[:2]
        
        # Centrar el cuadrado de referencia
        center_x = w // 2
        center_y = h // 2
        
        # Posicionar la esquina superior izquierda del cuadrado
        ref_x = center_x - (self.tile_size // 2)
        ref_y = center_y - (self.tile_size // 2)
        
        self.reference_position = (ref_x, ref_y)
        
        print(f"[INFO] Cuadrado de referencia en: ({ref_x}, {ref_y})")
        print(f"[INFO] Tamaño de ficha: {self.tile_size}px")
    
    def extract_tile(self, x, y):
        """
        Extrae una ficha de la imagen en la posición x, y
        """
        h, w = self.image.shape[:2]
        
        # Verificar que esté dentro de los límites
        if x < 0 or y < 0 or x + self.tile_size > w or y + self.tile_size > h:
            return None
        
        tile = self.image[y:y+self.tile_size, x:x+self.tile_size]
        return tile
    
    def create_board_matrix(self):
        """
        Crea la matriz del tablero completo
        """
        if self.reference_position is None:
            self.detect_reference_square()
        
        h, w = self.image.shape[:2]
        ref_x, ref_y = self.reference_position
        
        # Calcular dimensiones de la matriz
        cols_right = (w - ref_x) // self.tile_size
        cols_left = ref_x // self.tile_size
        rows_down = (h - ref_y) // self.tile_size
        rows_up = ref_y // self.tile_size
        
        total_cols = cols_left + cols_right
        total_rows = rows_up + rows_down
        
        print(f"[INFO] Matriz detectada: {total_rows}x{total_cols}")
        
        # Crear matriz vacía
        board_matrix = []
        
        # Recorrer desde arriba hacia abajo
        start_y = ref_y - (rows_up * self.tile_size)
        
        for row in range(total_rows):
            current_row = []
            start_x = ref_x - (cols_left * self.tile_size)
            
            for col in range(total_cols):
                x = start_x + (col * self.tile_size)
                y = start_y + (row * self.tile_size)
                
                tile_img = self.extract_tile(x, y)
                
                if tile_img is not None:
                    tile_name, rotation, score = self.matcher.match_tile(tile_img)
                    
                    if tile_name:
                        current_row.append({
                            'tile': tile_name,
                            'rotation': rotation,
                            'confidence': round(score, 2)
                        })
                    else:
                        current_row.append(None)  # Espacio vacío
                else:
                    current_row.append(None)
            
            board_matrix.append(current_row)
        
        return board_matrix
    
    def generate_debug_image(self):
        """
        Genera imagen con la cuadrícula dibujada
        """
        if self.reference_position is None:
            self.detect_reference_square()
        
        debug_img = draw_grid(self.image, self.tile_size, self.reference_position)
        
        if DEBUG_MODE:
            save_debug_image(debug_img)
        
        return debug_img