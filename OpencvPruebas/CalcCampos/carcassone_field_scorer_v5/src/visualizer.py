"""Visualización de resultados.

Este módulo genera las imágenes con los contornos de campos y un resumen
con la puntuación por campo y los totales por jugador.
"""
import numpy as np
import cv2
from typing import List, Dict
from src.field_detector import Field
from config.colors import PLAYER_NAMES
import random


class FieldVisualizer:
    """Visualizador de campos y resumen de puntuaciones."""

    def __init__(self, original_image: np.ndarray):
        """
        Inicializa el visualizador.

        Args:
            original_image: Imagen original del tablero
        """
        self.original = original_image.copy()
        self.height, self.width = original_image.shape[:2]

    def generate_field_colors(self, num_fields: int) -> Dict[int, tuple]:
        """Genera colores aleatorios deterministas para cada campo."""
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
        """Calcula el centroide visual para colocar la etiqueta del campo."""
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
        """Dibuja contornos y etiquetas de campos según su dueño."""
        result_image = self.original.copy()
        owner_colors = {
            'MEEPLE_1': (200, 0, 200),
            'MEEPLE_2': (50, 50, 50),
            'TIE': (255, 255, 0),
            None: (150, 150, 150)
        }
        for field in fields:
            field_mask = (field.pixels * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            owner = field_results[field.id]['owner']
            # normalizar empate
            owner_key = 'TIE' if owner == 'TIE' else owner
            color = owner_colors.get(owner_key, (150, 150, 150))
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
        """Genera una imagen con la lista de campos y los totales por jugador."""
        img_height = max(200, 100 + len(field_results) * 30 + 80)
        summary = np.ones((img_height, 600, 3), dtype=np.uint8) * 255
        y_pos = 30
        cv2.putText(summary, "RESULTADOS DE CAMPOS", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y_pos += 40
        if len(field_results) == 0:
            cv2.putText(summary, "No se detectaron campos", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
            y_pos += 30
        else:
            for field_id, data in sorted(field_results.items()):
                owner_name = PLAYER_NAMES.get(data['owner'], 'Sin dueño')
                if data['is_tie']:
                    owner_name = 'EMPATE'

                complete = data.get('castles_complete', data.get('castles', 0))
                incomplete = data.get('castles_incomplete', 0)
                if incomplete > 0:
                    castles_text = f"{complete} completos + {incomplete} incompletos"
                else:
                    castles_text = f"{complete} completos"

                text = f"Campo {field_id}: {owner_name} | {data['score']} pts | {castles_text}"
                cv2.putText(summary, text, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_pos += 25
        y_pos += 15
        cv2.putText(summary, "TOTALES:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 25
        for player_id in ['MEEPLE_1', 'MEEPLE_2']:
            player_name = PLAYER_NAMES.get(player_id, player_id)
            total = player_totals.get(player_id, 0)
            text = f"{player_name}: {total} puntos"
            cv2.putText(summary, text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 25
        return summary