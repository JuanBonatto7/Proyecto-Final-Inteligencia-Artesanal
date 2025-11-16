"""
Visualización de resultados (CORREGIDO - etiquetas en posición correcta).
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
    
    def find_label_position(self, field_pixels: np.ndarray, original_image: np.ndarray) -> tuple:
        """
        Encuentra la mejor posición para la etiqueta del campo.
        CORREGIDO: Usa solo píxeles verdes/amarillos del campo, evita castillos naranjas.
        
        Args:
            field_pixels: Máscara binaria del campo
            original_image: Imagen original (RGB)
            
        Returns:
            (cx, cy): Coordenadas del centroide en área del campo
        """
        # Obtener coordenadas de todos los píxeles del campo
        y_coords, x_coords = np.where(field_pixels)
        
        if len(y_coords) == 0:
            return None, None
        
        # CLAVE: Filtrar píxeles que NO son naranjas (castillos)
        # Aceptamos verdes y amarillos, rechazamos naranjas
        field_color_pixels = []
        
        for y, x in zip(y_coords, x_coords):
            r, g, b = original_image[y, x]
            
            # Detectar si es naranja (castillo): r alto, g medio, b bajo
            is_orange = (r > 200 and g > 80 and g < 180 and b < 100)
            
            # Si NO es naranja, es parte del campo
            if not is_orange:
                field_color_pixels.append((x, y))
        
        if not field_color_pixels:
            # Fallback: usar centroide normal si no hay píxeles de campo
            return int(x_coords.mean()), int(y_coords.mean())
        
        # Calcular centroide solo de píxeles del campo (no castillos)
        field_x = [p[0] for p in field_color_pixels]
        field_y = [p[1] for p in field_color_pixels]
        
        cx = int(np.mean(field_x))
        cy = int(np.mean(field_y))
        
        return cx, cy
    
    def draw_field_boundaries(
        self, 
        fields: List[Field],
        field_results: Dict[int, Dict]
    ) -> np.ndarray:
        """
        Dibuja contornos de campos con colores según el dueño.
        CORREGIDO: Etiquetas se colocan en área verde, no en castillos.
        
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
            
            # CORREGIDO: Calcular posición de etiqueta usando solo píxeles verdes
            cx, cy = self.find_label_position(field.pixels, self.original)
            
            if cx is not None and cy is not None:
                text = f"F{field.id}"
                
                # Fondo blanco para mejor legibilidad
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Rectángulo de fondo
                cv2.rectangle(
                    result_image,
                    (cx - text_width//2 - 3, cy - text_height//2 - 3),
                    (cx + text_width//2 + 3, cy + text_height//2 + baseline + 3),
                    (255, 255, 255),
                    -1
                )
                
                # Texto
                cv2.putText(
                    result_image, text, 
                    (cx - text_width//2, cy + text_height//2),
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
        img_height = max(200, 100 + len(field_results) * 30 + 80)
        summary = np.ones((img_height, 600, 3), dtype=np.uint8) * 255
        
        y_pos = 30
        
        # Título
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
        
        # Mostrar SIEMPRE ambos jugadores, incluso con 0 puntos
        for player_id in ['MEEPLE_1', 'MEEPLE_2']:
            player_name = PLAYER_NAMES.get(player_id, player_id)
            total = player_totals.get(player_id, 0)
            text = f"{player_name}: {total} puntos"
            cv2.putText(
                summary, text, (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
            y_pos += 25
        
        return summary
    
    def visualize_all_castles(self, castle_analyzer, fields: List[Field], scorer) -> np.ndarray:
        """
        Crea una visualización de todos los castillos en el tablero.
        
        Args:
            castle_analyzer: Analizador de castillos
            fields: Lista de campos
            scorer: Calculador de puntuación
            
        Returns:
            Imagen con castillos visualizados
        """
        img = self.original.copy()
        
        # Obtener máscaras
        complete_mask = castle_analyzer.get_complete_castles_mask()
        incomplete_mask = castle_analyzer.get_incomplete_castles_mask()
        
        # Dibujar castillos completos en verde
        complete_contours, _ = cv2.findContours(
            complete_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, complete_contours, -1, (0, 255, 0), 3)
        
        # Dibujar castillos incompletos en rojo
        incomplete_contours, _ = cv2.findContours(
            incomplete_mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, incomplete_contours, -1, (255, 0, 0), 3)
        
        # Etiquetar cada castillo completo
        for castle_id in castle_analyzer.complete_castles:
            castle_pixels = (castle_analyzer.labeled_castles == castle_id)
            y_coords, x_coords = np.where(castle_pixels)
            if len(y_coords) > 0:
                cy, cx = int(y_coords.mean()), int(x_coords.mean())
                
                # Fondo blanco
                text = f"C{castle_id}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (cx-tw//2-2, cy-th//2-2), 
                             (cx+tw//2+2, cy+th//2+2), (255, 255, 255), -1)
                
                # Texto verde
                cv2.putText(img, text, (cx-tw//2, cy+th//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        
        # Etiquetar cada castillo incompleto
        for castle_id in castle_analyzer.incomplete_castles:
            castle_pixels = (castle_analyzer.labeled_castles == castle_id)
            y_coords, x_coords = np.where(castle_pixels)
            if len(y_coords) > 0:
                cy, cx = int(y_coords.mean()), int(x_coords.mean())
                
                # Fondo blanco
                text = f"I{castle_id}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (cx-tw//2-2, cy-th//2-2), 
                             (cx+tw//2+2, cy+th//2+2), (255, 255, 255), -1)
                
                # Texto rojo
                cv2.putText(img, text, (cx-tw//2, cy+th//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
        
        # Agregar leyenda
        legend_y = 30
        cv2.putText(img, "CASTILLOS:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        legend_y += 30
        cv2.putText(img, "Verde (C#) = Completos (suman puntos)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        legend_y += 25
        cv2.putText(img, "Rojo (I#) = Incompletos (NO suman)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
        
        # Agregar estadísticas
        stats = castle_analyzer.get_castle_statistics()
        legend_y += 35
        cv2.putText(img, f"Total: {stats['total_castles']} castillos", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        legend_y += 25
        cv2.putText(img, f"Completos: {stats['complete_castles']}", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
        legend_y += 25
        cv2.putText(img, f"Incompletos: {stats['incomplete_castles']}", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
        
        return img
    
    def visualize_meeples(
        self, 
        meeple_masks: Dict[str, np.ndarray],
        meeple_validity: Dict[str, Dict[str, List[np.ndarray]]] = None
    ) -> np.ndarray:
        """
        Visualiza la posición de los meeples en el tablero.
        Muestra meeples válidos con círculos y meeples inválidos con cruces rojas.
        
        Args:
            meeple_masks: Diccionario con máscaras de meeples por jugador
            meeple_validity: Diccionario con clasificación de meeples válidos/inválidos
            
        Returns:
            Imagen con meeples marcados
        """
        img = self.original.copy()
        
        if meeple_validity is None:
            # Modo antiguo: solo mostrar las máscaras sin clasificación
            for player, mask in meeple_masks.items():
                if player == 'MEEPLE_1':
                    color = (255, 0, 255)  # Magenta
                    label = 'J1'
                elif player == 'MEEPLE_2':
                    color = (0, 0, 255)  # Azul
                    label = 'J2'
                else:
                    continue
                
                # Encontrar posiciones de meeples
                y_coords, x_coords = np.where(mask)
                
                for y, x in zip(y_coords, x_coords):
                    cv2.circle(img, (x, y), 8, color, -1)
                    cv2.putText(img, label, (x-10, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Modo nuevo: mostrar válidos e inválidos diferenciados
            for player, validity_data in meeple_validity.items():
                if player == 'MEEPLE_1':
                    valid_color = (255, 0, 255)  # Magenta
                    label = 'J1'
                elif player == 'MEEPLE_2':
                    valid_color = (0, 0, 255)  # Azul
                    label = 'J2'
                else:
                    continue
                
                # Dibujar meeples VÁLIDOS con círculos
                for meeple_mask in validity_data['valid']:
                    y_coords, x_coords = np.where(meeple_mask)
                    if len(y_coords) > 0:
                        # Centro del meeple
                        cy, cx = int(y_coords.mean()), int(x_coords.mean())
                        cv2.circle(img, (cx, cy), 12, valid_color, 2)
                        cv2.putText(img, label, (cx-10, cy-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Dibujar meeples INVÁLIDOS con cruces rojas
                for meeple_mask in validity_data['invalid']:
                    y_coords, x_coords = np.where(meeple_mask)
                    if len(y_coords) > 0:
                        # Centro del meeple
                        cy, cx = int(y_coords.mean()), int(x_coords.mean())
                        # Dibujar cruz roja
                        size = 15
                        cv2.line(img, (cx-size, cy-size), (cx+size, cy+size), (0, 0, 255), 3)
                        cv2.line(img, (cx-size, cy+size), (cx+size, cy-size), (0, 0, 255), 3)
                        cv2.putText(img, f"{label}-X", (cx-15, cy-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Agregar leyenda
        legend_y = 30
        cv2.putText(img, "DEBUG MEEPLES", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        legend_y += 25
        cv2.putText(img, "Circulo Magenta (J1) = Meeple Valido Jugador 1", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        legend_y += 25
        cv2.putText(img, "Circulo Azul (J2) = Meeple Valido Jugador 2", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        legend_y += 25
        cv2.putText(img, "Cruz Roja (X) = Meeple Invalido (toca camino o no toca campo)", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img