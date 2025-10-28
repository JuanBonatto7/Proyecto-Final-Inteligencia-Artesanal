"""
Visualización de resultados.
"""
import numpy as np
import cv2
from typing import List, Dict
from src.field_detector import Field
from config.colors import PLAYER_NAMES
import random


class FieldVisualizer:
    """Visualiza campos y resultados."""

    def __init__(self, original_image: np.ndarray):
        """
        Inicializa el visualizador.

        Args:
            original_image: Imagen original del tablero
        """
        self.original = original_image.copy()
        self.height, self.width = original_image.shape[:2]

    def generate_field_colors(self, num_fields: int) -> Dict[int, tuple]:
        """
        Genera colores únicos para cada campo.

        Args:
            num_fields: Número de campos

        Returns:
            Diccionario con colores únicos por campo
        """
        colors = {}
        random.seed(42)
        for i in range(1, num_fields + 1):
            colors[i] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        return colors

    def find_label_position(self, field_pixels: np.ndarray, original_image: np.ndarray) -> tuple:
        """
        Encuentra la mejor posición para la etiqueta del campo.

        Args:
            field_pixels: Máscara binaria del campo
            original_image: Imagen original (RGB)

        Returns:
            (cx, cy): Coordenadas del centroide en área del campo
        """
        y_coords, x_coords = np.where(field_pixels)
        if len(y_coords) == 0:
            return None, None
        field_color_pixels = []
        for y, x in zip(y_coords, x_coords):
            r, g, b = original_image[y, x]
            is_orange = (r > 200 and g > 80 and g < 180 and b < 100)
            if not is_orange:
                field_color_pixels.append((x, y))
        if not field_color_pixels:
            return int(x_coords.mean()), int(y_coords.mean())
        field_x = [p[0] for p in field_color_pixels]
        field_y = [p[1] for p in field_color_pixels]
        cx = int(np.mean(field_x))
        cy = int(np.mean(field_y))
        return cx, cy

    def draw_field_boundaries(self, fields: List[Field], field_results: Dict[int, Dict]) -> np.ndarray:
        """
        Dibuja contornos de campos con colores según el dueño.

        Args:
            fields: Lista de campos
            field_results: Resultados de puntuación

        Returns:
            Imagen con contornos dibujados
        """
        result_image = self.original.copy()
        owner_colors = {
            'meeple_1': (200, 0, 200),
            'meeple_2': (50, 50, 50),
            'tie': (255, 255, 0),
            None: (150, 150, 150)
        }
        for field in fields:
            field_mask = (field.pixels * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            owner = field_results[field.id]['owner']
            color = owner_colors.get(owner, (150, 150, 150))
            cv2.drawContours(result_image, contours, -1, color, 3)
            cx, cy = self.find_label_position(field.pixels, self.original)
            if cx is not None and cy is not None:
                text = f"F{field.id}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    result_image,
                    (cx - text_width//2 - 3, cy - text_height//2 - 3),
                    (cx + text_width//2 + 3, cy + text_height//2 + baseline + 3),
                    (255, 255, 255),
                    -1
                )
                cv2.putText(
                    result_image, text,
                    (cx - text_width//2, cy + text_height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        return result_image

    def create_summary_image(self, field_results: Dict[int, Dict], player_totals: Dict[str, int]) -> np.ndarray:
        """
        Crea imagen resumen con información de puntuación.

        Args:
            field_results: Resultados por campo
            player_totals: Totales por jugador

        Returns:
            Imagen resumen
        """
        img_height = max(200, 100 + len(field_results) * 30 + 80)
        summary = np.ones((img_height, 600, 3), dtype=np.uint8) * 255
        y_pos = 30
        cv2.putText(
            summary, "RESULTADOS DE CAMPOS", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        y_pos += 40
        if len(field_results) == 0:
            cv2.putText(
                summary, "No se detectaron campos", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1
            )
            y_pos += 30
        else:
            for field_id, data in sorted(field_results.items()):
                owner_name = PLAYER_NAMES.get(data['owner'], 'Sin dueno')
                if data['is_tie']:
                    owner_name = 'EMPATE'
                text = f"Campo {field_id}: {owner_name} | {data['score']} pts | {data['castles']} castillos"
                cv2.putText(
                    summary, text, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )
                y_pos += 25
        y_pos += 15
        cv2.putText(
            summary, "TOTALES:", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        y_pos += 25
        for player_id in ['meeple_1', 'meeple_2']:
            player_name = PLAYER_NAMES.get(player_id, player_id)
            total = player_totals.get(player_id, 0)
            text = f"{player_name}: {total} puntos"
            cv2.putText(
                summary, text, (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
            y_pos += 25
        return summary