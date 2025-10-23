"""
Paso 9: Clasificación de fichas (MEJORADO).
Soporta plantillas sin rotación en el nombre.
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
    
    Soporta dos formatos de plantillas:
    1. Con rotación: A_0.png, A_1.png, A_2.png, A_3.png
    2. Sin rotación: A.png (detecta rotación automáticamente)
    """
    
    def __init__(
        self,
        templates_path: str = Config.TILES_TEMPLATE_PATH,
        feature_type: str = Config.FEATURE_DESCRIPTOR,
        match_threshold: float = Config.MATCH_RATIO_THRESHOLD,
        min_matches: int = Config.MIN_MATCHES_THRESHOLD
    ):
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
            return cv2.ORB_create(nfeatures=1000)
        elif self.feature_type == "SIFT":
            return cv2.SIFT_create()
        elif self.feature_type == "AKAZE":
            return cv2.AKAZE_create()
        else:
            print(f"⚠ Tipo de descriptor '{self.feature_type}' no soportado, usando ORB")
            return cv2.ORB_create(nfeatures=1000)
    
    def _load_templates(self) -> Dict[str, List[Dict]]:
        """
        Carga las plantillas de fichas desde disco.
        Soporta dos formatos: A.png o A_0.png
        
        Returns:
            Diccionario {tile_type: [template_data, ...]}
        """
        templates = {}
        
        if not os.path.exists(self.templates_path):
            print(f"⚠ Advertencia: No existe el directorio de plantillas: {self.templates_path}")
            return templates
        
        # Primero, intentar cargar plantillas con rotación explícita (A_0.png)
        templates_with_rotation = self._load_templates_with_rotation()
        
        # Segundo, cargar plantillas sin rotación (A.png) y generar rotaciones
        templates_without_rotation = self._load_templates_without_rotation()
        
        # Combinar ambos tipos
        for tile_type in TILE_TYPES:
            tile_templates = []
            
            # Priorizar plantillas con rotación explícita
            if tile_type in templates_with_rotation:
                tile_templates.extend(templates_with_rotation[tile_type])
            elif tile_type in templates_without_rotation:
                tile_templates.extend(templates_without_rotation[tile_type])
            
            if tile_templates:
                templates[tile_type] = tile_templates
        
        print(f"  ✓ Plantillas cargadas: {len(templates)} tipos de fichas")
        total_templates = sum(len(v) for v in templates.values())
        print(f"  ✓ Total de variaciones: {total_templates}")
        
        return templates
    
    def _load_templates_with_rotation(self) -> Dict[str, List[Dict]]:
        """Carga plantillas con formato A_0.png, A_1.png, etc."""
        templates = {}
        
        for tile_type in TILE_TYPES:
            tile_templates = []
            
            for rotation in ROTATIONS:
                # Buscar archivo con rotación
                template_file = os.path.join(
                    self.templates_path,
                    f"{tile_type}_{rotation}.png"
                )
                
                if os.path.exists(template_file):
                    img = cv2.imread(template_file)
                    if img is not None:
                        template_data = self._process_template(img, tile_type, rotation)
                        if template_data is not None:
                            tile_templates.append(template_data)
            
            if tile_templates:
                templates[tile_type] = tile_templates
        
        return templates
    
    def _load_templates_without_rotation(self) -> Dict[str, List[Dict]]:
        """
        Carga plantillas con formato A.png y genera las 4 rotaciones.
        """
        templates = {}
        
        for tile_type in TILE_TYPES:
            # Buscar archivo sin rotación
            template_file = os.path.join(self.templates_path, f"{tile_type}.png")
            
            if os.path.exists(template_file):
                img = cv2.imread(template_file)
                if img is not None:
                    tile_templates = []
                    
                    # Generar las 4 rotaciones
                    for rotation in ROTATIONS:
                        # Rotar imagen
                        rotated_img = self._rotate_image(img, rotation)
                        
                        # Procesar plantilla
                        template_data = self._process_template(rotated_img, tile_type, rotation)
                        if template_data is not None:
                            tile_templates.append(template_data)
                    
                    if tile_templates:
                        templates[tile_type] = tile_templates
        
        return templates
    
    def _rotate_image(self, image: np.ndarray, rotation: int) -> np.ndarray:
        """
        Rota una imagen según el código de rotación.
        
        Args:
            image: Imagen original
            rotation: 0=0°, 1=90°, 2=180°, 3=270°
            
        Returns:
            Imagen rotada
        """
        if rotation == 0:
            return image.copy()
        elif rotation == 1:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 2:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == 3:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image.copy()
    
    def _process_template(
        self,
        img: np.ndarray,
        tile_type: str,
        rotation: int
    ) -> Optional[Dict]:
        """
        Procesa una plantilla y extrae sus características.
        
        Returns:
            Diccionario con datos de la plantilla o None si falla
        """
        # Convertir a escala de grises si es necesario
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Detectar características
        kp, desc = self.detector.detectAndCompute(gray, None)
        
        if desc is None or len(kp) < self.min_matches:
            return None
        
        return {
            'image': img,
            'gray': gray,
            'keypoints': kp,
            'descriptors': desc,
            'rotation': rotation,
            'tile_type': tile_type
        }
    
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
        
        print(f"  Clasificando {len(tile_images)} fichas...")
        
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
            
            # Mostrar progreso cada 10 fichas
            if (idx + 1) % 10 == 0:
                print(f"  Progreso: {idx + 1}/{len(tile_images)}")
        
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
        
        # Asegurar que la imagen tenga un tamaño mínimo
        if tile_image.shape[0] < 20 or tile_image.shape[1] < 20:
            return 'A', 0, 0.0
        
        # Convertir a escala de grises
        if len(tile_image.shape) == 3:
            gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile_image
        
        # Detectar características en la ficha
        kp, desc = self.detector.detectAndCompute(gray, None)
        
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
                
                try:
                    # Matching con k=2 para ratio test
                    matches = matcher.knnMatch(desc, template['descriptors'], k=2)
                    
                    # Ratio test (Lowe's ratio test)
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < self.match_threshold * n.distance:
                                good_matches.append(m)
                    
                    # Calcular score
                    num_good = len(good_matches)
                    if num_good >= self.min_matches:
                        # Score ponderado por número de matches y distancia promedio
                        avg_distance = np.mean([m.distance for m in good_matches])
                        score = num_good / (1 + avg_distance / 100)
                        
                        if score > best_score:
                            best_score = score
                            best_match = template
                
                except cv2.error as e:
                    # Silenciosamente ignorar errores de matching
                    continue
        
        if best_match is None:
            return 'A', 0, 0.0
        
        # Calcular confianza normalizada (0-1)
        confidence = min(best_score / 50.0, 1.0)
        
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
        
        # Contar tipos de fichas
        tile_counts = {}
        confidence_sum = 0
        
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
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                
                cv2.rectangle(
                    result,
                    (center[0] - tw // 2 - 5, center[1] - th - 15),
                    (center[0] + tw // 2 + 5, center[1] + 15),
                    (255, 255, 255),
                    -1
                )
                
                # Texto principal
                cv2.putText(
                    result, label,
                    (center[0] - tw // 2, center[1]),
                    font, 0.6, (0, 0, 0), 2
                )
                
                # Confianza
                cv2.putText(
                    result, conf_label,
                    (center[0] - tw // 2, center[1] + 15),
                    font, 0.4, color, 1
                )
            
            # Estadísticas
            tile_counts[tile.tile_type] = tile_counts.get(tile.tile_type, 0) + 1
            confidence_sum += tile.confidence
        
        # Panel de estadísticas
        avg_conf = confidence_sum / len(tiles) if tiles else 0
        y_offset = 30
        cv2.putText(
            result,
            f"Tiles: {len(tiles)} | Avg Conf: {avg_conf:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        # Mostrar distribución de tipos
        y_offset = 60
        for tile_type, count in sorted(tile_counts.items()):
            cv2.putText(
                result,
                f"{tile_type}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 20
        
        return result