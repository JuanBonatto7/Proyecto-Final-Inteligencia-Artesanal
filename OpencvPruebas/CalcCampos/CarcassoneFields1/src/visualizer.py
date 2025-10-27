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
        """Genera colores únicos para cada campo."""
        colors = {}
        random.seed(42)  # Para reproducibilidad
        
        for i in range(1, num_fields + 1):
            colors[i] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        
        return colors
    
    def draw_field_boundaries(
        self, 
        fields: List[Field],
        field_results: Dict[int, Dict]
    ) -> np.ndarray:
        """
        Dibuja contornos de campos con colores según el dueño.
        
        Args:
            fields: Lista de campos
            field_results: Resultados de puntuación
            
        Returns:
            Imagen con contornos dibujados
        """
        result_image = self.original.copy()
        
        # Colores para cada jugador
        owner_colors = {
            'MEEPLE_1': (200, 0, 200),  # Violeta
            'MEEPLE_2': (50, 50, 50),   # Gris oscuro
            'TIE': (255, 255, 0),        # Amarillo
            None: (150, 150, 150)        # Gris
        }
        
        for field in fields:
            # Crear máscara uint8 para findContours
            field_mask = (field.pixels * 255).astype(np.uint8)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(
                field_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Color según dueño
            owner = field_results[field.id]['owner']
            color = owner_colors.get(owner, (150, 150, 150))
            
            # Dibujar contorno
            cv2.drawContours(result_image, contours, -1, color, 3)
            
            # Añadir etiqueta
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    text = f"F{field.id}"
                    cv2.putText(
                        result_image, text, (cx-20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
        
        return result_image
    
    def create_summary_image(
        self,
        field_results: Dict[int, Dict],
        player_totals: Dict[str, int]
    ) -> np.ndarray:
        """
        Crea imagen resumen con información de puntuación.
        
        Args:
            field_results: Resultados por campo
            player_totals: Totales por jugador
            
        Returns:
            Imagen resumen
        """
        # Crear imagen blanca
        img_height = max(200, 100 + len(field_results) * 30)
        summary = np.ones((img_height, 600, 3), dtype=np.uint8) * 255
        
        y_pos = 30
        
        # Título - CAMBIO AQUÍ: FONT_HERSHEY_SIMPLEX en lugar de BOLD
        cv2.putText(
            summary, "RESULTADOS DE CAMPOS", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        y_pos += 40
        
        # Resultados por campo
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
        
        # Totales
        y_pos += 15
        cv2.putText(
            summary, "TOTALES:", (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        y_pos += 25
        
        if len(player_totals) == 0:
            cv2.putText(
                summary, "Sin puntos", (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1
            )
        else:
            for player, total in player_totals.items():
                player_name = PLAYER_NAMES.get(player, player)
                text = f"{player_name}: {total} puntos"
                cv2.putText(
                    summary, text, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )
                y_pos += 25
        
        return summary