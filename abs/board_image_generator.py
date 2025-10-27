from PIL import Image, ImageDraw
import os
from typing import Tuple, List, Optional, Dict

from origin_matrix import Board, Tile

__all__ = ["BoardImageGenerator"]

class BoardImageGenerator:
    """Generador de imágenes de tableros de Carcassonne."""

    FILE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    # Posiciones de meeple (1..9): 1=arriba-izquierda, 5=centro, 9=abajo-derecha
    MEEPLE_POSITIONS: Dict[int, Tuple[float, float]] = {
        1: (0.20, 0.20),
        2: (0.50, 0.15),
        3: (0.80, 0.20),
        4: (0.15, 0.50),
        5: (0.50, 0.50),
        6: (0.85, 0.50),
        7: (0.20, 0.80),
        8: (0.50, 0.85),
        9: (0.80, 0.80),
    }

    def __init__(self, tiles_folder: str, tile_size: int = 200):
        self.tiles_folder = tiles_folder
        self.tile_size = tile_size
        self.meeple_colors: Dict[int, Tuple[int, int, int]] = {
            1: (163, 73, 164),
            2: (0, 0, 0),
        }
        self._tile_cache: Dict[str, Image.Image] = {}

    def _find_tile_file(self, tile_name: str) -> Optional[str]:
        """Busca el archivo de imagen de la loseta por nombre."""
        for extension in self.FILE_EXTENSIONS:
            tile_path = os.path.join(self.tiles_folder, f"{tile_name}{extension}")
            if os.path.exists(tile_path):
                return tile_path
        return None

    def _create_placeholder_tile(self, tile_name: str) -> Image.Image:
        """Crea una imagen placeholder si falta el asset del tile."""
        img = Image.new('RGB', (self.tile_size, self.tile_size), color='lightgray')
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, self.tile_size - 1, self.tile_size - 1], outline='black', width=3)
        bbox = draw.textbbox((0, 0), tile_name)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.tile_size - text_width) // 2
        y = (self.tile_size - text_height) // 2
        draw.text((x, y), tile_name, fill='black')
        return img

    def load_tile_image(self, tile_name: str) -> Image.Image:
        """Carga la imagen del tile desde disco o cache; usa placeholder si no existe."""
        if tile_name in self._tile_cache:
            return self._tile_cache[tile_name].copy()

        tile_path = self._find_tile_file(tile_name)
        if tile_path is None:
            img = self._create_placeholder_tile(tile_name)
        else:
            try:
                img = Image.open(tile_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = self._resize_and_center(img)
            except Exception:
                img = self._create_placeholder_tile(tile_name)

        self._tile_cache[tile_name] = img.copy()
        return img

    def _resize_and_center(self, img: Image.Image) -> Image.Image:
        """Redimensiona la imagen preservando proporción y la centra en un canvas cuadrado."""
        width, height = img.size
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = self.tile_size
            new_height = int(self.tile_size / aspect_ratio)
        else:
            new_height = self.tile_size
            new_width = int(self.tile_size * aspect_ratio)

        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if new_width != self.tile_size or new_height != self.tile_size:
            canvas = Image.new('RGB', (self.tile_size, self.tile_size), color='white')
            offset_x = (self.tile_size - new_width) // 2
            offset_y = (self.tile_size - new_height) // 2
            canvas.paste(img_resized, (offset_x, offset_y))
            return canvas

        return img_resized

    def rotate_tile(self, img: Image.Image, rotation_degrees: int) -> Image.Image:
        """Rota en grados: 0, 90, 180, 270."""
        return img.rotate(-rotation_degrees, expand=False)

    def get_meeple_position(self, position: int) -> Tuple[int, int]:
        """Convierte la posición 1..9 a coordenadas de píxel dentro del tile."""
        rel_x, rel_y = self.MEEPLE_POSITIONS.get(position, (0.5, 0.5))
        x = int(rel_x * self.tile_size)
        y = int(rel_y * self.tile_size)
        return x, y

    def draw_meeple(self, img: Image.Image, player: int, position: int) -> Image.Image:
        """Dibuja un meeple simple sobre la imagen del tile."""
        if player <= 0:
            return img
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        x, y = self.get_meeple_position(position)
        radius = self.tile_size // 10
        color = self.meeple_colors.get(player, (128, 128, 128))

        # Cabeza
        head_radius = radius // 2
        draw.ellipse(
            [x - head_radius, y - radius, x + head_radius, y - radius + head_radius * 2],
            fill=color, outline='black', width=1
        )
        # Cuerpo
        body_width = radius
        body_height = radius
        draw.ellipse(
            [x - body_width, y - radius // 3, x + body_width, y + body_height],
            fill=color, outline='black', width=2
        )
        return img_copy

    def generate_board_image(self, board: Board, output_path: str = "tablero.jpg", verbose: bool = False) -> Image.Image:
        """Genera y guarda la imagen del tablero completo."""
        rows = len(board.board)
        cols = len(board.board[0]) if rows > 0 else 0
        board_width = cols * self.tile_size
        board_height = rows * self.tile_size
        board_img = Image.new('RGB', (board_width, board_height), color='white')

        for i, row in enumerate(board.board):
            for j, tile in enumerate(row):
                if tile is None:
                    continue
                tile_img = self.load_tile_image(tile.type)
                tile_img = self.rotate_tile(tile_img, tile.rotation)
                if tile.meeple_info is not None:
                    tile_img = self.draw_meeple(tile_img, tile.meeple_info[0], tile.meeple_info[1])
                x = j * self.tile_size
                y = i * self.tile_size
                board_img.paste(tile_img, (x, y))

        board_img.save(output_path, 'JPEG', quality=95)

        if verbose:
            print("✓ Tablero generado exitosamente")
            print(f"  Archivo: {output_path}")
            print(f"  Dimensiones: {board_width}x{board_height} píxeles")
            print(f"  Tablero: {rows}x{cols} losetas")
            print(f"  Tiles en cache: {len(self._tile_cache)}")

        return board_img

    def add_player_color(self, player_number: int, color: Tuple[int, int, int]):
        """Permite configurar el color de un jugador para el meeple."""
        self.meeple_colors[player_number] = color

if __name__ == "__main__":
    # Ejemplo mínimo actualizado a rotación en grados y meeple=None
    sample_tiles = [
        [Tile("W", 0, None), Tile("D", 90, None)],
        [Tile("C", 0, None), Tile("L", 0, (1, 5))]
    ]
    board = Board(board=sample_tiles)
    gen = BoardImageGenerator(tiles_folder="./tiles", tile_size=200)
    gen.generate_board_image(board, "tablero_ejemplo.jpg", verbose=True)