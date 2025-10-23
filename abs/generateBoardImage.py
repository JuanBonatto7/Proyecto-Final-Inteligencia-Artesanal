from PIL import Image, ImageDraw
import os
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Tile:
    """Representa una loseta del tablero de Carcassonne"""
    nombre: str  # Letra del tile: A, B, C, D, E, F, G, H, I, J, K, M, N, O, P, Q, R, S, T, U, V, W, X, Y
    orientacion: int  # 0, 1, 2, 3 (rotación en múltiplos de 90°)
    meeple: Tuple[int, int]  # (jugador, posición) 
                              # jugador: 0=ninguno, 1=jugador1, 2=jugador2
                              # posición: 0-8 (ver mapa de posiciones abajo)





@dataclass
class Board:
    """Representa el tablero completo del juego"""
    tiles: List[List[Tile]]  # matriz de losetas






class BoardImageGenerator:
    def __init__(self, tiles_folder: str, tile_size: int = 200):
        """
        Inicializa el generador de imágenes del tablero de Carcassonne

        Args:
            tiles_folder: ruta a la carpeta con las imágenes de las losetas (A.jpg, B.jpg, etc.)
            tile_size: tamaño de cada loseta en píxeles (recomendado: 150-250)
        """
        self.tiles_folder = tiles_folder
        self.tile_size = tile_size
        self.meeple_colors = {
            1: (255, 0, 0),    # Rojo para jugador 1
            2: (0, 0, 255)     # Azul para jugador 2
        }
        
    def load_tile_image(self, tile_name: str) -> Image.Image:
        """Carga la imagen de una loseta desde el disco"""
        # Intentar con .jpg primero, luego .png
        tile_path = os.path.join(self.tiles_folder, f"{tile_name}.jpg")
        if not os.path.exists(tile_path):
            tile_path = os.path.join(self.tiles_folder, f"{tile_name}.png")
        
        if not os.path.exists(tile_path):
            # Si no existe la imagen, crear una loseta vacía con el nombre
            img = Image.new('RGB', (self.tile_size, self.tile_size), color='lightgray')
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, self.tile_size-1, self.tile_size-1], outline='black', width=2)
            # Dibujar el nombre del tile en el centro
            text_size = self.tile_size // 3
            draw.text((self.tile_size//2 - text_size//4, self.tile_size//2 - text_size//4), 
                     tile_name, fill='black')
            return img
        
        img = Image.open(tile_path)
        img = img.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
        return img
    
    def rotate_tile(self, img: Image.Image, orientation: int) -> Image.Image:
        """
        Rota la imagen según la orientación
        0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
        """
        angle = orientation * 90
        return img.rotate(-angle, expand=False)
    
    def get_meeple_position(self, position: int) -> Tuple[int, int]:
        """
        Calcula las coordenadas del meeple según su posición (0-8)
        Mapa de posiciones para Carcassonne:
        
        0 (NO)  1 (N)   2 (NE)
        3 (O)   4 (C)   5 (E)
        6 (SO)  7 (S)   8 (SE)
        
        Uso típico:
        - Posición 4 (Centro): Monasterios, campos centrales
        - Posiciones 1,3,5,7 (Bordes): Caminos, ciudades en bordes
        - Posiciones 0,2,6,8 (Esquinas): Campos en esquinas
        """
        # Posiciones relativas en la loseta (0.0 a 1.0)
        positions_map = {
            0: (0.20, 0.20),  # Noroeste
            1: (0.50, 0.15),  # Norte
            2: (0.80, 0.20),  # Noreste
            3: (0.15, 0.50),  # Oeste
            4: (0.50, 0.50),  # Centro
            5: (0.85, 0.50),  # Este
            6: (0.20, 0.80),  # Suroeste
            7: (0.50, 0.85),  # Sur
            8: (0.80, 0.80),  # Sureste
        }
        
        rel_x, rel_y = positions_map.get(position, (0.5, 0.5))
        x = int(rel_x * self.tile_size)
        y = int(rel_y * self.tile_size)
        
        return (x, y)
    
    def draw_meeple(self, img: Image.Image, player: int, position: int) -> Image.Image:
        """Dibuja un meeple en la loseta (estilo Carcassonne)"""
        if player == 0:
            return img
        
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        x, y = self.get_meeple_position(position)
        radius = self.tile_size // 10  # Tamaño del meeple
        color = self.meeple_colors.get(player, (0, 0, 0))
        
        # Dibujar meeple simplificado
        # Cabeza
        head_radius = radius // 2
        draw.ellipse([x - head_radius, y - radius, x + head_radius, y - radius + head_radius*2], 
                     fill=color, outline='black', width=1)
        
        # Cuerpo
        body_width = radius
        body_height = radius
        draw.ellipse([x - body_width, y - radius//3, x + body_width, y + body_height], 
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
        print(f"✓ Imagen del tablero guardada en: {output_path}")
        print(f"  Dimensiones: {board_width}x{board_height} píxeles")
        print(f"  Tablero: {rows}x{cols} losetas")
        return board_img


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    print("Generador de Tablero de Carcassonne con Pillow")
    print("=" * 50)
    
    # Ejemplo 1: Tablero pequeño 3x3
    ejemplo_tiles = [
        [
            Tile("D", 0, (1, 4)),   # Tile D sin rotar, meeple rojo en centro
            Tile("U", 1, (0, 0)),   # Tile U rotado 90°, sin meeple
            Tile("B", 0, (2, 1))    # Tile B, meeple azul en norte
        ],
        [
            Tile("V", 2, (0, 0)),   # Tile V rotado 180°, sin meeple
            Tile("N", 0, (1, 5)),   # Tile N, meeple rojo en este
            Tile("D", 3, (0, 0))    # Tile D rotado 270°, sin meeple
        ],
        [
            Tile("U", 0, (2, 7)),   # Tile U, meeple azul en sur
            Tile("V", 1, (0, 0)),   # Tile V rotado 90°, sin meeple
            Tile("D", 2, (1, 3))    # Tile D rotado 180°, meeple rojo en oeste
        ]
    ]
    
    board = Board(tiles=ejemplo_tiles)
    
    # IMPORTANTE: Ajustar la ruta a tu carpeta de tiles
    generator = BoardImageGenerator(
        tiles_folder="/dataset/Tiles",  # Carpeta donde están A.jpg, B.jpg, C.jpg, etc.
        tile_size=200            # Tamaño de cada loseta (150-250 recomendado)
    )
    
    print("\nGenerando imagen del tablero...")
    generator.generate_board_image(board, "tablero_carcassonne.jpg")
    
    print("\n" + "=" * 50)
    print("Mapa de posiciones de meeples:")
    print("  0 (NO)  1 (N)   2 (NE)")
    print("  3 (O)   4 (C)   5 (E)")
    print("  6 (SO)  7 (S)   8 (SE)")
    print("\nColores:")
    print("  Jugador 1: Rojo")
    print("  Jugador 2: Azul")