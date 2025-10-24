"""
Paso 8: Extraer tiles usando homografía.
Basado en el proyecto original de Carcassonne tracking.
"""

import cv2
import numpy as np
from typing import Dict, Any, List
from pipeline.pipeline_step import PipelineStep
from config import Config


class FindTilesStep(PipelineStep):
    """
    Extrae tiles individuales del tablero usando la homografía.
    
    Para cada posición (i, j) en el rango de grilla:
    1. Calcula las 4 esquinas en coordenadas del tablero ideal
    2. Proyecta esas esquinas a la imagen usando H_inv
    3. Aplica transformación de perspectiva para "unwarp" el tile
    4. Filtra tiles vacíos por densidad de bordes
    """
    
    def __init__(self):
        super().__init__("Find Tiles")
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae tiles usando homografía"""
        
        H = inputs.get('homography')
        H_inv = inputs.get('homography_inv')
        img = inputs['img']
        
        if H is None or H_inv is None:
            print("⚠ Sin homografía, no se pueden extraer tiles")
            inputs['tile_regions'] = []
            inputs['tile_images'] = []
            inputs['tile_positions'] = []
            inputs['debug_image'] = img.copy()
            return inputs
        
        print(f"  Extrayendo tiles en rango [{Config.GRID_RANGE_MIN}, {Config.GRID_RANGE_MAX}]...")
        
        tiles = []
        tile_images = []
        tile_positions = []
        h, w = img.shape[:2]
        
        # Iterar sobre grilla
        for i in range(Config.GRID_RANGE_MIN, Config.GRID_RANGE_MAX + 1):
            for j in range(Config.GRID_RANGE_MIN, Config.GRID_RANGE_MAX + 1):
                
                # Calcular esquinas en coordenadas de tablero
                tl_board = np.array([j * Config.TILE_SIZE, i * Config.TILE_SIZE], dtype=np.float32)
                tr_board = np.array([(j+1) * Config.TILE_SIZE, i * Config.TILE_SIZE], dtype=np.float32)
                br_board = np.array([(j+1) * Config.TILE_SIZE, (i+1) * Config.TILE_SIZE], dtype=np.float32)
                bl_board = np.array([j * Config.TILE_SIZE, (i+1) * Config.TILE_SIZE], dtype=np.float32)
                
                # Proyectar a imagen
                corners_board = np.array([tl_board, tr_board, br_board, bl_board], dtype=np.float32)
                corners_img = cv2.perspectiveTransform(corners_board.reshape(1, -1, 2), H_inv)
                corners_img = corners_img.reshape(-1, 2)
                
                # Validar que estén dentro de la imagen
                if not self._are_points_valid(corners_img, w, h):
                    continue
                
                # Extraer tile con transformación de perspectiva
                tile_img = self._unwarp_tile(img, corners_img)
                
                if tile_img is None:
                    continue
                
                # Verificar que no sea tile vacío
                if not self._is_valid_tile(tile_img):
                    continue
                
                tiles.append({
                    'img': tile_img,
                    'pos': (i, j),
                    'corners_img': corners_img
                })
                tile_images.append(tile_img)
                tile_positions.append((i, j))
        
        print(f"  ✅ {len(tiles)} tiles extraídos")
        
        # Guardar en formato compatible con el pipeline
        inputs['tile_regions'] = [(t['corners_img'], t['img']) for t in tiles]
        inputs['tile_images'] = tile_images
        inputs['tile_positions'] = tile_positions
        inputs['tiles'] = tiles  # Formato completo para clasificación
        
        # Visualización
        debug_image = self._create_visualization(img, tiles)
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _are_points_valid(self, points: np.ndarray, width: int, height: int) -> bool:
        """Verifica que todos los puntos estén dentro de la imagen"""
        for point in points:
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                return False
        return True
    
    def _unwarp_tile(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Extrae tile aplicando transformación de perspectiva.
        
        Args:
            img: Imagen original
            corners: 4 esquinas del tile en imagen [tl, tr, br, bl]
            
        Returns:
            Tile normalizado de tamaño TILE_SIZE x TILE_SIZE
        """
        # Puntos destino (cuadrado perfecto)
        dst_points = np.array([
            [0, 0],
            [Config.TILE_SIZE, 0],
            [Config.TILE_SIZE, Config.TILE_SIZE],
            [0, Config.TILE_SIZE]
        ], dtype=np.float32)
        
        # Calcular transformación
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_points)
        
        # Aplicar transformación
        tile = cv2.warpPerspective(img, M, (Config.TILE_SIZE, Config.TILE_SIZE))
        
        return tile
    
    def _is_valid_tile(self, tile_img: np.ndarray) -> bool:
        """
        Verifica que el tile no esté vacío usando densidad de bordes.
        
        Args:
            tile_img: Imagen del tile
            
        Returns:
            True si el tile parece contener una loseta real
        """
        # Convertir a gris si es necesario
        if len(tile_img.shape) == 3:
            gray = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = tile_img
        
        # Detectar bordes
        edges = cv2.Canny(gray, 50, 150)
        
        # Calcular densidad de bordes
        edge_density = np.mean(edges)
        
        # Si la densidad es muy baja, es probablemente fondo vacío
        return edge_density >= Config.MIN_EDGE_DENSITY
    
    def _create_visualization(self, img: np.ndarray, tiles: List[Dict]) -> np.ndarray:
        """Visualiza tiles extraídos"""
        
        if len(img.shape) == 2:
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            result = img.copy()
        
        # Dibujar bounding boxes de tiles extraídos
        for idx, tile_data in enumerate(tiles):
            corners = tile_data['corners_img']
            i, j = tile_data['pos']
            
            # Dibujar polígono
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Etiquetar con posición
            center = corners.mean(axis=0).astype(int)
            cv2.putText(
                result, f"({i},{j})", tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
            )
        
        # Info
        cv2.putText(
            result, f"Tiles: {len(tiles)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        
        return result