# ============================================================================
# utils/visualization.py
# ============================================================================

import os
import cv2
import numpy as np
from typing import List

from CalcCampos.carcassonne_scorer.models.field import Field

from CalcCampos.carcassonne_scorer.models.game_state import GameState
from CalcCampos.carcassonne_scorer.processors.castle_detector import CastleDetector
from CalcCampos.carcassonne_scorer.processors.image_processor import ImageProcessor
from CalcCampos.carcassonne_scorer.utils.mask_utils import MaskUtils
from config.settings import Config

class Visualizer:
    """Generador de visualizaciones del proceso."""
    
    def __init__(self, output_dir: str = Config.OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_masks(self, processor: ImageProcessor):
        """Guarda todas las máscaras detectadas."""
        for name, mask in processor.masks.items():
            path = os.path.join(self.output_dir, f"01_mask_{name.lower()}.png")
            cv2.imwrite(path, mask)
        
        print(f"✓ Máscaras guardadas en {self.output_dir}")
    
    def save_castle_analysis(self, processor: ImageProcessor, castle_detector: CastleDetector):
        """Visualiza castillos cerrados vs abiertos."""
        vis = processor.image.copy()
        castle_mask = processor.get_mask('CASTLE')
        empty_mask = processor.get_mask('EMPTY')
        
        # Obtener componentes de castillos
        num_castles, castle_labels = MaskUtils.get_connected_components(castle_mask)
        
        for castle_id in range(1, num_castles):
            single_castle = (castle_labels == castle_id).astype(np.uint8) * 255
            
            # Determinar si está cerrado
            is_closed = castle_detector.is_castle_closed(single_castle, empty_mask)
            
            # Color: verde si cerrado, rojo si abierto
            color = (0, 255, 0) if is_closed else (0, 0, 255)
            
            # Encontrar contorno
            contours, _ = cv2.findContours(single_castle, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)
            
            # Añadir etiqueta
            moments = cv2.moments(single_castle)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                text = "CERRADO" if is_closed else "ABIERTO"
                cv2.putText(vis, text, (cx-35, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        
        path = os.path.join(self.output_dir, "02_castles_analysis.png")
        cv2.imwrite(path, vis)
        print(f"✓ Análisis de castillos guardado en {path}")
    
    def save_fields_visualization(self, 
                                   image: np.ndarray,
                                   fields: List[Field]):
        """Crea una visualización de los campos detectados."""
        vis = image.copy()
        
        # Crear imagen de campos coloreados
        fields_colored = np.zeros_like(image)
        
        for i, field in enumerate(fields):
            # Color aleatorio para cada campo
            color = np.random.randint(50, 255, 3).tolist()
            fields_colored[field.mask > 0] = color
        
        # Combinar con imagen original
        alpha = 0.5
        vis = cv2.addWeighted(vis, 1-alpha, fields_colored, alpha, 0)
        
        # Añadir etiquetas
        for field in fields:
            # Encontrar centroide
            moments = cv2.moments(field.mask)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # Texto
                text = f"F{field.id}"
                cv2.putText(vis, text, (cx-10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        path = os.path.join(self.output_dir, "03_fields_detected.png")
        cv2.imwrite(path, vis)
        
        print(f"✓ Visualización de campos guardada en {path}")
    
    def save_final_visualization(self,
                                  image: np.ndarray,
                                  game_state: GameState):
        """Crea visualización final con puntuaciones."""
        vis = image.copy()
        
        # Dibujar información de cada campo
        for field in game_state.fields:
            if not field.is_valid or field.owner is None:
                continue
            
            # Encontrar contorno
            contours, _ = cv2.findContours(field.mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Color según dueño
            if field.owner == 1:
                color = (163, 73, 164)  # Violeta
            elif field.owner == 2:
                color = (0, 0, 0)  # Negro
            else:  # Empate
                color = (255, 255, 0)  # Amarillo (más visible que cyan)
            
            # Dibujar contorno
            cv2.drawContours(vis, contours, -1, color, 3)
            
            # Añadir información
            moments = cv2.moments(field.mask)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                owner_text = f"P{field.owner}" if field.owner != 0 else "TIE"
                text = f"{owner_text}: {field.points}pts"
                
                cv2.putText(vis, text, (cx-30, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Añadir puntuación total
        text1 = f"P1: {game_state.total_scores[1]} puntos"
        text2 = f"P2: {game_state.total_scores[2]} puntos"
        
        cv2.putText(vis, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (163, 73, 164), 2)
        cv2.putText(vis, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(vis, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        path = os.path.join(self.output_dir, "04_final_scores.png")
        cv2.imwrite(path, vis)
        
        print(f"✓ Visualización final guardada en {path}")