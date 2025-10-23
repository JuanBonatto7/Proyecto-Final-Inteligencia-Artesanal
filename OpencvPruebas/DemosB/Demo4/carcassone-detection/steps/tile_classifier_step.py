"""
Paso 9: Clasificación de fichas.
"""

import cv2
import numpy as np
import os
from typing import Dict, Any, List, Tuple, Optional
from pipeline.pipeline_step import PipelineStep
from config import Config, TILE_TYPES, ROTATIONS
from models.tile import Tile


class TileClassifierStep(PipelineStep):
    """
    Clasifica cada ficha detectada identificando su tipo y rotación.
    
    Utiliza descriptores de características (ORB/SIFT) para comparar
    con plantillas de fichas conocidas.
    """
    
    def __init__(
        self,
        templates_path: str = Config.TILES_TEMPLATE_PATH,
        feature_type: str = Config.FEATURE_DESCRIPTOR,
        match_threshold: float = Config.MATCH_RATIO_THRESHOLD,
        min_matches: int = Config.MIN_MATCHES_THRESHOLD
    ):
        """
        Inicializa el clasificador de fichas.
        
        Args:
            templates_path: Ruta a las plantillas de fichas
            feature_type: Tipo de descriptor ("ORB", "SIFT", "AKAZE")
            match_threshold: Umbral para ratio test de matches
            min_matches: Mínimo número de matches para considerar válido
        """
        super().__init__("Tile Classifier")
        self.templates_path = templates_path
        self.feature_type = feature_type
        self.match_threshold = match_threshold
        self.min_matches = min_matches
        
        # Crear detector de características
        self.detector = self._create_feature_detector()
        
        # Cargar plantillas
        self.templates = self._load_templates()
    
    def _create_feature_detector(self):
        """Crea el detector de características según configuración."""
        if self.feature_type == "ORB":
            return cv2.ORB_create(nfeatures=500)
        elif self.feature_type == "SIFT":
            return cv2.SIFT_create()
        elif self.feature_type == "AKAZE":
            return cv2.AKAZE_create()
        else:
            print(f"⚠ Tipo de descriptor '{self.feature_type}' no soportado, usando ORB")
            return cv2.ORB_create(nfeatures=500)
    
    def _load_templates(self) -> Dict[str, List[Dict]]:
        """
        Carga las plantillas de fichas desde disco.
        
        Returns:
            Diccionario {tile_type: [template_data, ...]}
        """
        templates = {}
        
        if not os.path.exists(self.templates_path):
            print(f"⚠ Advertencia: No existe el directorio de plantillas: {self.templates_path}")
            print("  Creando plantillas de ejemplo...")
            self._create_example_templates()
        
        # Intentar cargar plantillas
        for tile_type in TILE_TYPES:
            tile_templates = []
            
            for rotation in ROTATIONS:
                template_file = os.path.join(
                    self.templates_path,
                    f"{tile_type}_{rotation}.png"
                )
                
                if os.path.exists(template_file):
                    img = cv2.imread(template_file)
                    if img is not None:
                        # Detectar características en la plantilla
                        kp, desc = self.detector.detectAndCompute(img, None)
                        
                        tile_templates.append({
                            'image': img,
                            'keypoints': kp,
                            'descriptors': desc,
                            'rotation': rotation,
                            'tile_type': tile_type
                        })
            
            if tile_templates:
                templates[tile_type] = tile_templates
        
        print(f"  Plantillas cargadas: {len(templates)} tipos de fichas")
        return templates
    
    def _create_example_templates(self):
        """Crea plantillas de ejemplo para demostración."""
        os.makedirs(self.templates_path, exist_ok=True)
        
        # Crear algunas plantillas sintéticas simples
        for i, tile_type in enumerate(TILE_TYPES[:5]):  # Solo primeras 5
            for rotation in ROTATIONS:
                # Crear imagen de plantilla sintética
                img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                
                # Dibujar patrón único por tipo
                color = (i * 50 % 255, (i * 30) % 255, (i * 70) % 255)
                
                if rotation == 0:
                    cv2.rectangle(img, (20, 20), (80, 40), color, -1)
                elif rotation == 1:
                    cv2.rectangle(img, (60, 20), (80, 80), color, -1)
                elif rotation == 2:
                    cv2.rectangle(img, (20, 60), (80, 80), color, -1)
                else:
                    cv2.rectangle(img, (20, 20), (40, 80), color, -1)
                
                # Agregar texto
                cv2.putText(img, tile_type, (35, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                # Guardar
                filename = os.path.join(self.templates_path, f"{tile_type}_{rotation}.png")
                cv2.imwrite(filename, img)
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clasifica todas las fichas detectadas.
        
        Args:
            inputs: Debe contener 'tile_images', 'tile_positions', 'tile_regions'
            
        Returns:
            Inputs actualizado con:
                - 'classified_tiles': Lista de objetos Tile
                - 'debug_image': Visualización
        """
        tile_images = inputs.get('tile_images', [])
        tile_positions = inputs.get('tile_positions', [])
        tile_regions = inputs.get('tile_regions', [])
        
        if not tile_images:
            print("⚠ No hay fichas para clasificar")
            inputs['classified_tiles'] = []
            inputs['debug_image'] = inputs.get('img', np.zeros((100, 100, 3)))
            return inputs
        
        classified_tiles = []
        
        for idx, (tile_img, position) in enumerate(zip(tile_images, tile_positions)):
            # Clasificar ficha
            tile_type, rotation, confidence = self._classify_tile(tile_img)
            
            # Obtener corners si están disponibles
            corners = None
            if idx < len(tile_regions):
                corners = tile_regions[idx][0]
            
            # Crear objeto Tile
            tile = Tile(
                tile_type=tile_type,
                rotation=rotation,
                position=position,
                corners=corners,
                confidence=confidence,
                image=tile_img
            )
            
            classified_tiles.append(tile)
        
        inputs['classified_tiles'] = classified_tiles
        
        # Crear visualización
        original_img = inputs.get('img_warped', inputs.get('img'))
        debug_image = self._create_visualization(original_img, classified_tiles)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _classify_tile(
        self,
        tile_image: np.ndarray
    ) -> Tuple[str, int, float]:
        """
        Clasifica una ficha individual.
        
        Args:
            tile_image: Imagen de la ficha a clasificar
            
        Returns:
            Tupla (tile_type, rotation, confidence)
        """
        if not self.templates:
            # Sin plantillas, retornar clasificación por defecto
            return 'A', 0, 0.0
        
        # Detectar características en la ficha
        kp, desc = self.detector.detectAndCompute(tile_image, None)
        
        if desc is None or len(kp) < self.min_matches:
            return 'A', 0, 0.0
        
        # Crear matcher
        if self.feature_type == "ORB":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        best_match = None
        best_score = 0
        
        # Comparar con todas las plantillas
        for tile_type, templates in self.templates.items():
            for template in templates:
                if template['descriptors'] is None:
                    continue
                
                # Matching
                matches = matcher.knnMatch(desc, template['descriptors'], k=2)
                
                # Ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.match_threshold * n.distance:
                            good_matches.append(m)
                
                # Calcular score
                if len(good_matches) > self.min_matches:
                    score = len(good_matches)
                    
                    if score > best_score:
                        best_score = score
                        best_match = template
        
        if best_match is None:
            return 'A', 0, 0.0
        
        # Calcular confianza normalizada
        confidence = min(best_score / 100.0, 1.0)
        
        return best_match['tile_type'], best_match['rotation'], confidence
    
    def _create_visualization(
        self,
        image: np.ndarray,
        tiles: List[Tile]
    ) -> np.ndarray:
        """Crea visualización con fichas clasificadas."""
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
        
        for tile in tiles:
            if tile.corners is not None:
                # Dibujar contorno
                points = tile.corners.astype(np.int32)
                
                # Color según confianza
                if tile.confidence > 0.7:
                    color = (0, 255, 0)  # Verde - alta confianza
                elif tile.confidence > 0.4:
                    color = (0, 255, 255)  # Amarillo - media confianza
                else:
                    color = (0, 0, 255)  # Rojo - baja confianza
                
                cv2.polylines(result, [points], True, color, 2)
                
                # Etiqueta
                center = tuple(np.mean(points, axis=0).astype(int))
                label = f"{tile.tile_type}_{tile.rotation}"
                conf_label = f"{tile.confidence:.2f}"
                
                # Fondo para texto
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.5, 2)
                
                cv2.rectangle(
                    result,
                    (center[0] - tw // 2 - 5, center[1] - th - 10),
                    (center[0] + tw // 2 + 5, center[1] + 10),
                    (255, 255, 255),
                    -1
                )
                
                # Texto
                cv2.putText(result, label, (center[0] - tw // 2, center[1]),
                           font, 0.5, (0, 0, 0), 2)
                cv2.putText(result, conf_label, (center[0] - tw // 2, center[1] + 15),
                           font, 0.4, (100, 100, 100), 1)
        
        # Estadísticas
        avg_conf = sum(t.confidence for t in tiles) / len(tiles) if tiles else 0
        cv2.putText(
            result,
            f"Tiles: {len(tiles)} | Avg Conf: {avg_conf:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        return result