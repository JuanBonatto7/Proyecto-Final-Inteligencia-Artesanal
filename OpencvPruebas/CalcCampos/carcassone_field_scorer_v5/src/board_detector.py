"""
Detección de límites del tablero de Carcassonne.
Identifica áreas blancas (fuera del tablero) vs área de juego.
"""
import numpy as np
import cv2
from scipy import ndimage


class BoardDetector:
    """Detecta los límites del tablero y áreas fuera de juego."""

    def __init__(self, image: np.ndarray, white_threshold: int = 200):
        """
        Inicializa el detector de tablero.

        Args:
            image: Imagen RGB del tablero
            white_threshold: Umbral para detectar blanco (0-255)
        """
        self.image = image
        self.white_threshold = white_threshold
        self.height, self.width = image.shape[:2]

    def detect_white_areas(self) -> np.ndarray:
        """
        Detecta áreas blancas (fuera del tablero).

        Returns:
            Máscara binaria donde True = blanco (fuera del tablero)
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        white_mask = gray > self.white_threshold
        kernel = np.ones((5, 5), dtype=np.uint8)
        white_mask = ndimage.binary_closing(white_mask, structure=kernel, iterations=2)
        white_mask = ndimage.binary_opening(white_mask, structure=kernel, iterations=1)
        return white_mask

    def create_board_mask(self) -> np.ndarray:
        """
        Crea máscara del área de juego (inverso del blanco).

        Returns:
            Máscara binaria donde True = dentro del tablero
        """
        white_areas = self.detect_white_areas()
        board_mask = ~white_areas
        return board_mask

    def detect_board_edges(self) -> np.ndarray:
        """
        Detecta los bordes del tablero (transición blanco -> tablero).

        Returns:
            Máscara binaria de los bordes del tablero
        """
        white_areas = self.detect_white_areas()
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated_white = ndimage.binary_dilation(white_areas, structure=kernel, iterations=1)
        edges = dilated_white & ~white_areas
        return edges

    def is_touching_white(self, mask: np.ndarray, expansion: int = 3) -> bool:
        """
        Verifica si una máscara toca áreas blancas.

        Args:
            mask: Máscara binaria del objeto a verificar
            expansion: Píxeles de expansión para detectar proximidad

        Returns:
            True si el objeto toca o está muy cerca del blanco
        """
        white_areas = self.detect_white_areas()
        kernel = np.ones((3, 3), dtype=np.uint8)
        expanded_mask = ndimage.binary_dilation(mask, structure=kernel, iterations=expansion)
        intersection = expanded_mask & white_areas
        return np.any(intersection)

    def filter_mask_by_board(self, mask: np.ndarray) -> np.ndarray:
        """
        Filtra una máscara para incluir solo píxeles dentro del tablero.

        Args:
            mask: Máscara a filtrar

        Returns:
            Máscara filtrada (solo área de juego)
        """
        board_mask = self.create_board_mask()
        filtered_mask = mask & board_mask
        return filtered_mask