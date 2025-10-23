from PIL import Image, ImageDraw
import os
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Tile:
    """Representa una loseta del tablero"""
    nombre: str  # nombre del archivo de imagen (sin extensión)
    orientacion: int  # 0, 1, 2, 3 (rotación en múltiplos de 90°)
    meeple: Tuple[int, int]  # (jugador, posición) donde jugador: 0=ninguno, 1=jugador1, 2=jugador2
                              # posición: 0-8 para las 9 posiciones posibles

@dataclass
class Board:
    """Representa el tablero completo"""
    tiles: List[List[Tile]]  # matriz de losetas

class BoardImageGenerator:
    def __init__(self, tiles_folder: str, tile_size: int = 100):
        """
        Inicializa el generador de imágenes del tablero
        
        Args:
            tiles_folder: ruta a la carpeta con las imágenes de las losetas
            tile_size: tamaño de cada loseta en píxeles (default: 100x100)
        """
        self.tiles_folder = tiles_folder
        self.tile_size = tile_size
        self.meeple_colors = {
            1: (255, 0, 0),    # Rojo para jugador 1
            2: (0, 0, 255)     # Azul para jugador 2
        }
        
    def load_tile_image(self, tile_name: str) -> Image.Image:
        """Carga la imagen de una loseta desde el disco"""
        tile_path = os.path.join(self.tiles_folder, f"{tile_name}.jpg")
        if not os.path.exists(tile_path):
            tile_path = os.path.join(self.tiles_folder, f"{tile_name}.png")
        
        if not os.path.exists(tile_path):
            # Si no existe la imagen, crear una loseta vacía
            img = Image.new('RGB', (self.tile_size, self.tile_size), color='white')
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, self.tile_size-1, self.tile_size-1], outline='black')
            draw.text((10, 40), tile_name, fill='black')
            return img
        
        img = Image.open(tile_path)
        img = img.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
        return img
    
    def rotate_tile(self, img: Image.Image, orientation: int) -> Image.Image:
        """Rota la imagen según la orientación (0, 1, 2, 3)"""
        angle = orientation * 90
        return img.rotate(-angle, expand=False)
    
    def get_meeple_position(self, position: int) -> Tuple[int, int]:
        """
        Calcula las coordenadas del meeple según su posición (0-8)
        Las posiciones se distribuyen en una cuadrícula 3x3:
        0 1 2
        3 4 5
        6 7 8
        """
        row = position // 3
        col = position % 3
        
        # Calcular coordenadas con margen
        margin = self.tile_size // 6
        spacing = self.tile_size // 3
        
        x = margin + col * spacing
        y = margin + row * spacing
        
        return (x, y)
    
    def draw_meeple(self, img: Image.Image, player: int, position: int) -> Image.Image:
        """Dibuja un meeple en la loseta"""
        if player == 0:
            return img
        
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        x, y = self.get_meeple_position(position)
        radius = self.tile_size // 12
        color = self.meeple_colors.get(player, (0, 0, 0))
        
        # Dibujar un círculo para representar el meeple
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                     fill=color, outline='black', width=2)
        
        return img_copy
    
    def generate_board_image(self, board: Board, output_path: str = "tablero.jpg"):
        """
        Genera la imagen del tablero completo
        
        Args:
            board: objeto Board con la matriz de losetas
            output_path: ruta donde guardar la imagen generada
        """
        rows = len(board.tiles)
        cols = len(board.tiles[0]) if rows > 0 else 0
        
        # Crear imagen del tamaño del tablero completo
        board_width = cols * self.tile_size
        board_height = rows * self.tile_size
        board_img = Image.new('RGB', (board_width, board_height), color='white')
        
        # Procesar cada loseta
        for i, row in enumerate(board.tiles):
            for j, tile in enumerate(row):
                if tile is None:
                    continue
                
                # Cargar imagen de la loseta
                tile_img = self.load_tile_image(tile.nombre)
                
                # Rotar según orientación
                tile_img = self.rotate_tile(tile_img, tile.orientacion)
                
                # Dibujar meeple si existe
                tile_img = self.draw_meeple(tile_img, tile.meeple[0], tile.meeple[1])
                
                # Pegar en el tablero
                x = j * self.tile_size
                y = i * self.tile_size
                board_img.paste(tile_img, (x, y))
        
        # Guardar imagen
        board_img.save(output_path, 'JPEG', quality=95)
        print(f"Imagen del tablero guardada en: {output_path}")
        return board_img


# Ejemplo de uso
if __name__ == "__main__":
    # Crear un tablero de ejemplo 3x3
    ejemplo_tiles = [
        [
            Tile("ciudad", 0, (1, 4)),  # Ciudad sin rotar, meeple jugador 1 en centro
            Tile("camino", 1, (0, 0)),  # Camino rotado 90°, sin meeple
            Tile("monasterio", 0, (2, 8))  # Monasterio, meeple jugador 2 en esquina
        ],
        [
            Tile("campo", 2, (0, 0)),
            Tile("cruce", 0, (1, 0)),
            Tile("ciudad", 3, (0, 0))
        ],
        [
            Tile("camino", 0, (2, 5)),
            Tile("campo", 1, (0, 0)),
            Tile("ciudad", 2, (1, 2))
        ]
    ]
    
    board = Board(tiles=ejemplo_tiles)
    
    # Generar imagen del tablero
    generator = BoardImageGenerator(
        tiles_folder="./tiles",  # Carpeta con las imágenes de las losetas
        tile_size=150  # Tamaño de cada loseta en píxeles
    )
    
    generator.generate_board_image(board, "tablero_juego.jpg")