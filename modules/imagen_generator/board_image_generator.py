"""Generador de imágenes visuales de tableros de Carcassonne."""
from PIL import Image, ImageDraw
import os
from typing import Tuple, Dict, Optional

from origin_matrix import Board

__all__ = ["BoardImageGenerator"]


class BoardImageGenerator:
    """Generador de imágenes de tableros de Carcassonne."""

    FILE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    # Posiciones de meeple (1..9): 1=arriba-izquierda, 5=centro, 9=abajo-derecha
    MEEPLE_POSITIONS: Dict[int, Tuple[float, float]] = {
        1: (0.20, 0.20), 2: (0.50, 0.15), 3: (0.80, 0.20),
        4: (0.15, 0.50), 5: (0.50, 0.50), 6: (0.85, 0.50),
        7: (0.20, 0.80), 8: (0.50, 0.85), 9: (0.80, 0.80),
    }

    def __init__(self, tiles_folder: str):
        self.tiles_folder = tiles_folder
        self.meeple_colors: Dict[int, Tuple[int, int, int]] = {
            1: (163, 73, 164),  # Morado
            2: (0, 0, 0),        # Negro
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
        img = Image.new('RGB', (200, 200), color='lightgray')
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 200 - 1, 200 - 1],
                      outline='black', width=3)

        bbox = draw.textbbox((0, 0), tile_name)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (200 - text_width) // 2
        y = (200 - text_height) // 2
        draw.text((x, y), tile_name, fill='black')
        return img

    def load_tile_image(self, tile_name: str) -> Image.Image:
        """Carga la imagen del tile desde disco o cache."""
        if tile_name in self._tile_cache:
            return self._tile_cache[tile_name].copy()

        tile_path = self._find_tile_file(tile_name)

        if tile_path is None:
            img = self._create_placeholder_tile(tile_name)
        else:
            try:
                img = Image.open(tile_path).convert('RGB')
                img = img.resize((200, 200), Image.LANCZOS)
            except Exception:
                img = self._create_placeholder_tile(tile_name)

        self._tile_cache[tile_name] = img.copy()
        return img

    def draw_meeple(self, img: Image.Image, player: int, position: int) -> Image.Image:
        """Dibuja un meeple simple sobre la imagen del tile."""
        if player <= 0:
            return img

        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)

        rel_x, rel_y = self.MEEPLE_POSITIONS.get(position, (0.5, 0.5))
        x = int(rel_x * 200)
        y = int(rel_y * 200)

        radius = 200 // 10
        color = self.meeple_colors.get(player, (128, 128, 128))

        # Cabeza
        head_radius = radius // 2
        draw.ellipse(
            [x - head_radius, y - radius,
             x + head_radius, y - radius + head_radius * 2],
            fill=color, outline='black', width=1
        )

        # Cuerpo
        draw.ellipse(
            [x - radius, y - radius // 3,
            x + radius, y + radius],
            fill=color, outline='black', width=2
        )
        return img_copy

    def generate_board_image(self, board: Board, output_path: str = "tablero.jpg") -> Image.Image:
        """Genera y guarda la imagen del tablero completo."""
        rows = len(board.board)
        cols = len(board.board[0]) if rows > 0 else 0

        border_size = 200

        board_width = cols * 200 + border_size * 2
        board_height = rows * 200 + border_size * 2


        board_img = Image.new('RGB', (board_width, board_height), color='white')

        for i, row in enumerate(board.board):
            for j, tile in enumerate(row):
                if tile is None:
                    continue

                # Cargar y rotar tile
                tile_img = self.load_tile_image(tile.type)
                tile_img = tile_img.rotate(-tile.rotation, expand=False)

                # Dibujar meeple si existe
                if tile.meeple_info is not None:
                    tile_img = self.draw_meeple(
                        tile_img,
                        tile.meeple_info[0],
                        tile.meeple_info[1]
                    )

                # Pegar en posición correcta
                x = (j * 200) + border_size
                y = (i * 200) + border_size
                board_img.paste(tile_img, (x, y))

        # Redimensionar a tamaño final
        board_img = board_img.resize((800, 800), Image.LANCZOS)
        board_img.save(output_path, 'JPEG', quality=95)

        return board_img
