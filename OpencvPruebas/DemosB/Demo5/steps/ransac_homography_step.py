"""
Paso 7: Calcular homografía usando RANSAC.
Basado en el proyecto original de Carcassonne tracking.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from pipeline.pipeline_step import PipelineStep
from config import Config


class RANSACHomographyStep(PipelineStep):
    """
    Calcula homografía que mapea:
    - Puntos imagen → Coordenadas tablero ideal
    
    Usa los labels (i,j) directamente:
    - board_point = (j * TILE_SIZE, i * TILE_SIZE)
    """
    
    def __init__(self, tile_size: int = 64):
        super().__init__("RANSAC Homography")
        self.tile_size = tile_size
        self.ransac_threshold = 5.0
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula homografía"""
        
        intersections = inputs.get('intersections', np.array([]))
        intersection_labels = inputs.get('intersection_labels', {})
        img = inputs['img']
        
        if len(intersections) < 4:
            print(f"⚠ Solo {len(intersections)} intersecciones (mínimo 4)")
            inputs['homography'] = None
            inputs['homography_inv'] = None
            inputs['debug_image'] = img.copy()
            return inputs
        
        print(f"  Calculando homografía con {len(intersections)} puntos...")
        
        # 1. Preparar puntos
        img_points = []
        board_points = []
        
        for point in intersections:
            x, y = point
            
            # Obtener label (i, j)
            if (x, y) not in intersection_labels:
                continue
            
            i, j = intersection_labels[(x, y)]
            
            # 🔧 CLAVE: usar labels directamente para coordenadas tablero
            img_points.append([x, y])
            board_points.append([j * self.tile_size, i * self.tile_size])
        
        if len(img_points) < 4:
            print("⚠ No hay suficientes puntos con labels")
            inputs['homography'] = None
            inputs['homography_inv'] = None
            inputs['debug_image'] = img.copy()
            return inputs
        
        img_points = np.array(img_points, dtype=np.float32)
        board_points = np.array(board_points, dtype=np.float32)
        
        print(f"  Puntos válidos: {len(img_points)}")
        print(f"  Rango imagen: X [{img_points[:,0].min():.0f}, {img_points[:,0].max():.0f}], "
              f"Y [{img_points[:,1].min():.0f}, {img_points[:,1].max():.0f}]")
        print(f"  Rango tablero: X [{board_points[:,0].min():.0f}, {board_points[:,0].max():.0f}], "
              f"Y [{board_points[:,1].min():.0f}, {board_points[:,1].max():.0f}]")
        
        # 2. Calcular homografía: board → img
        H_board_to_img, mask = cv2.findHomography(
            board_points,
            img_points,
            cv2.RANSAC,
            self.ransac_threshold
        )
        
        if H_board_to_img is None:
            print("⚠ No se pudo calcular homografía")
            inputs['homography'] = None
            inputs['homography_inv'] = None
            inputs['debug_image'] = img.copy()
            return inputs
        
        # 3. Calcular inversa: img → board
        H_img_to_board, _ = cv2.findHomography(
            img_points,
            board_points,
            cv2.RANSAC,
            self.ransac_threshold
        )
        
        inliers = int(np.sum(mask))
        ratio = inliers / len(img_points)
        
        print(f"  ✅ Homografía: {inliers}/{len(img_points)} inliers ({ratio:.1%})")
        
        if ratio < 0.5:
            print(f"  ⚠ Ratio de inliers bajo ({ratio:.1%})")
        
        # Guardar
        inputs['homography'] = H_img_to_board  # img → board
        inputs['homography_inv'] = H_board_to_img  # board → img
        
        # Visualización
        debug_image = self._create_visualization(
            img, img_points, board_points, mask, H_board_to_img
        )
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _create_visualization(
        self,
        img: np.ndarray,
        img_points: np.ndarray,
        board_points: np.ndarray,
        mask: np.ndarray,
        H: np.ndarray
    ) -> np.ndarray:
        """Visualiza homografía"""
        
        if len(img.shape) == 2:
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            result = img.copy()
        
        # Dibujar puntos originales (inliers en verde, outliers en rojo)
        for i, point in enumerate(img_points):
            x, y = map(int, point)
            if mask[i]:
                color = (0, 255, 0)  # Verde = inlier
                radius = 6
            else:
                color = (0, 0, 255)  # Rojo = outlier
                radius = 4
            cv2.circle(result, (x, y), radius, color, -1)
            cv2.circle(result, (x, y), radius + 2, (255, 255, 255), 1)
        
        # Proyectar grilla ideal
        min_i = int(board_points[:, 1].min() / self.tile_size) - 2
        max_i = int(board_points[:, 1].max() / self.tile_size) + 2
        min_j = int(board_points[:, 0].min() / self.tile_size) - 2
        max_j = int(board_points[:, 0].max() / self.tile_size) + 2
        
        h, w = img.shape[:2]
        
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                # Punto en tablero
                board_p = np.array([[j * self.tile_size, i * self.tile_size]], dtype=np.float32)
                
                # Proyectar a imagen
                img_p = cv2.perspectiveTransform(board_p.reshape(1, 1, 2), H)
                img_p = img_p.reshape(2)
                
                x, y = map(int, img_p)
                
                # Dibujar si está dentro
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(result, (x, y), 2, (255, 255, 0), -1)
        
        # Info
        inliers = int(np.sum(mask))
        cv2.putText(
            result, f"Homography: {inliers}/{len(img_points)} inliers",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        
        return result