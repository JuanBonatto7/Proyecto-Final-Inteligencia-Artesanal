"""
Paso 9: Clasificar tiles extraídos.
Identifica el tipo y rotación de cada loseta.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
from pipeline.pipeline_step import PipelineStep
from config import Config


class TileClassifierStep(PipelineStep):
    """
    Clasifica cada tile usando Template Matching (NCC).
    
    Para cada tile:
    1. Lo compara con templates de referencia en 4 rotaciones
    2. Encuentra el mejor match (letra + rotación)
    3. Calcula confianza del match
    """
    
    def __init__(self):
        super().__init__("Tile Classifier")
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, np.ndarray]:
        """Carga templates de referencia"""
        templates = {}
        
        template_path = Path(Config.TILES_TEMPLATE_PATH)
        
        if not template_path.exists():
            print(f"⚠ Carpeta de templates no existe: {Config.TILES_TEMPLATE_PATH}")
            return templates
        
        # Buscar archivos de imagen
        for img_file in template_path.glob("*.*"):
            if img_file.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                continue
            
            # Nombre del tile (letra)
            tile_name = img_file.stem
            
            # Cargar imagen
            img = cv2.imread(str(img_file))
            
            if img is None:
                continue
            
            # Redimensionar a TILE_SIZE
            img = cv2.resize(img, (Config.TILE_SIZE, Config.TILE_SIZE))
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            templates[tile_name] = gray
        
        print(f"  ✅ {len(templates)} templates cargados")
        return templates
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Clasifica todos los tiles"""
        
        tiles = inputs.get('tiles', [])
        
        if not tiles:
            print("⚠ No hay tiles para clasificar")
            inputs['classified_tiles'] = []
            inputs['debug_image'] = inputs.get('img', np.zeros((600, 600, 3))).copy()
            return inputs
        
        if not self.templates:
            print("⚠ No hay templates de referencia")
            inputs['classified_tiles'] = []
            inputs['debug_image'] = inputs.get('img', np.zeros((600, 600, 3))).copy()
            return inputs
        
        print(f"  Clasificando {len(tiles)} tiles...")
        
        classified = []
        
        for tile_data in tiles:
            tile_img = tile_data['img']
            position = tile_data['pos']
            
            # Clasificar
            tile_type, rotation, confidence = self._classify_tile(tile_img)
            
            classified.append({
                'tile_type': tile_type,
                'rotation': rotation,
                'confidence': confidence,
                'position': position,
                'image': tile_img
            })
        
        print(f"  ✅ {len(classified)} tiles clasificados")
        
        # Mostrar algunos ejemplos
        high_conf = [t for t in classified if t['confidence'] >= Config.MIN_MATCH_SCORE]
        print(f"     Alta confianza (>={Config.MIN_MATCH_SCORE}): {len(high_conf)}/{len(classified)}")
        
        inputs['classified_tiles'] = classified
        
        # Visualización
        debug_image = self._create_visualization(inputs.get('img'), classified)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _classify_tile(self, tile_img: np.ndarray) -> Tuple[str, int, float]:
        """
        Clasifica un tile usando Template Matching.
        
        Args:
            tile_img: Imagen del tile
            
        Returns:
            Tupla (tile_type, rotation, confidence)
        """
        # Convertir a gris
        if len(tile_img.shape) == 3:
            gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile_img
        
        # Redimensionar si es necesario
        if gray.shape != (Config.TILE_SIZE, Config.TILE_SIZE):
            gray = cv2.resize(gray, (Config.TILE_SIZE, Config.TILE_SIZE))
        
        best_match = {
            'tile_type': '?',
            'rotation': 0,
            'score': -1.0
        }
        
        # Probar cada template
        for tile_name, template in self.templates.items():
            
            # Probar 4 rotaciones
            for rot in range(4):
                # Rotar template
                template_rot = np.rot90(template, rot)
                
                # Template Matching (NCC)
                result = cv2.matchTemplate(gray, template_rot, cv2.TM_CCOEFF_NORMED)
                score = np.max(result)
                
                if score > best_match['score']:
                    best_match = {
                        'tile_type': tile_name,
                        'rotation': rot,
                        'score': score
                    }
        
        # Si el score es muy bajo, marcar como desconocido
        if best_match['score'] < Config.MIN_MATCH_SCORE:
            return ('?', 0, best_match['score'])
        
        return (best_match['tile_type'], best_match['rotation'], best_match['score'])
    
    def _create_visualization(
        self,
        img: np.ndarray,
        classified_tiles: List[Dict]
    ) -> np.ndarray:
        """Visualiza tiles clasificados"""
        
        if img is None:
            result = np.zeros((600, 800, 3), dtype=np.uint8)
        else:
            if len(img.shape) == 2:
                result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                result = img.copy()
        
        # Crear panel con información
        panel_width = 300
        h, w = result.shape[:2]
        
        # Extender imagen para añadir panel lateral
        extended = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
        extended[:, :w] = result
        
        # Panel de información
        y_offset = 30
        cv2.putText(
            extended, "CLASIFICACION",
            (w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
        
        y_offset += 40
        
        # Mostrar algunos tiles clasificados
        for i, tile in enumerate(classified_tiles[:15]):  # Primeros 15
            i_pos, j_pos = tile['position']
            tile_type = tile['tile_type']
            rotation = tile['rotation']
            confidence = tile['confidence']
            
            color = (0, 255, 0) if confidence >= Config.MIN_MATCH_SCORE else (0, 165, 255)
            
            text = f"({i_pos},{j_pos}): {tile_type}R{rotation} {confidence:.0%}"
            cv2.putText(
                extended, text,
                (w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
            )
            y_offset += 20
            
            if y_offset > h - 30:
                break
        
        if len(classified_tiles) > 15:
            cv2.putText(
                extended, f"... +{len(classified_tiles) - 15} mas",
                (w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1
            )
        
        return extended