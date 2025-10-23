"""
Paso 7: Corrección de perspectiva con RANSAC (VERSIÓN MEJORADA).
Usa los puntos de intersección para calcular la homografía.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pipeline.pipeline_step import PipelineStep
from config import Config


class RANSACHomographyStep(PipelineStep):
    """
    Calcula y aplica corrección de perspectiva usando las intersecciones detectadas.
    Crea una grilla rectangular perfecta a partir de los puntos detectados.
    """
    
    def __init__(
        self,
        threshold: float = Config.RANSAC_THRESHOLD,
        max_iterations: int = Config.RANSAC_MAX_ITERATIONS,
        min_points: int = 4
    ):
        super().__init__("RANSAC Homography")
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.min_points = min_points
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula homografía y corrige perspectiva del tablero.
        
        Args:
            inputs: Debe contener 'intersections' y 'img'
            
        Returns:
            Inputs actualizado con:
                - 'homography_matrix': Matriz de homografía 3x3
                - 'img_warped': Imagen con perspectiva corregida
                - 'warped_points': Puntos transformados
                - 'grid_structure': Estructura de la grilla organizada
                - 'debug_image': Visualización
        """
        intersections = inputs.get('intersections', np.array([]))
        intersection_labels = inputs.get('intersection_labels', {})
        image = inputs['img']
        
        if len(intersections) < self.min_points:
            print(f"⚠ Advertencia: Solo {len(intersections)} intersecciones detectadas (mínimo: {self.min_points})")
            inputs['homography_matrix'] = None
            inputs['img_warped'] = image.copy()
            inputs['warped_points'] = []
            inputs['grid_structure'] = None
            inputs['debug_image'] = image.copy()
            return inputs
        
        print(f"  Procesando {len(intersections)} intersecciones...")
        
        # Organizar puntos en estructura de grilla
        grid_structure = self._organize_grid(intersections, intersection_labels)
        
        if grid_structure is None or len(grid_structure['points']) < self.min_points:
            print("⚠ No se pudo organizar la grilla correctamente")
            inputs['homography_matrix'] = None
            inputs['img_warped'] = image.copy()
            inputs['warped_points'] = intersections.tolist()
            inputs['grid_structure'] = None
            inputs['debug_image'] = image.copy()
            return inputs
        
        # Crear puntos destino (grilla perfecta)
        src_points, dst_points, board_size = self._create_destination_grid(grid_structure)
        
        if src_points is None or len(src_points) < self.min_points:
            print("⚠ No se pudieron crear puntos de destino")
            inputs['homography_matrix'] = None
            inputs['img_warped'] = image.copy()
            inputs['warped_points'] = intersections.tolist()
            inputs['grid_structure'] = grid_structure
            inputs['debug_image'] = image.copy()
            return inputs
        
        # Calcular homografía
        H, mask = cv2.findHomography(
            src_points,
            dst_points,
            cv2.RANSAC,
            self.threshold,
            maxIters=self.max_iterations
        )
        
        if H is None:
            print("⚠ No se pudo calcular la homografía")
            inputs['homography_matrix'] = None
            inputs['img_warped'] = image.copy()
            inputs['warped_points'] = intersections.tolist()
            inputs['grid_structure'] = grid_structure
            inputs['debug_image'] = image.copy()
            return inputs
        
        # Aplicar transformación
        warped = cv2.warpPerspective(image, H, board_size)
        
        # Transformar todos los puntos
        if len(intersections) > 0:
            intersections_reshaped = intersections.reshape(-1, 1, 2).astype(np.float32)
            warped_points = cv2.perspectiveTransform(intersections_reshaped, H)
            warped_points = warped_points.reshape(-1, 2).tolist()
        else:
            warped_points = []
        
        # Calcular estadísticas de la homografía
        inliers = np.sum(mask) if mask is not None else 0
        print(f"  Homografía calculada: {inliers}/{len(src_points)} inliers")
        
        inputs['homography_matrix'] = H
        inputs['img_warped'] = warped
        inputs['warped_points'] = warped_points
        inputs['grid_structure'] = grid_structure
        
        # Crear visualización
        debug_image = self._create_visualization(
            image, warped, intersections, src_points, dst_points, mask
        )
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _organize_grid(
        self,
        intersections: np.ndarray,
        intersection_labels: dict
    ) -> Optional[dict]:
        """
        Organiza las intersecciones en una estructura de grilla.
        
        Args:
            intersections: Puntos de intersección
            intersection_labels: Etiquetas (i, j) de cada intersección
            
        Returns:
            Diccionario con estructura de grilla o None
        """
        if len(intersections) == 0:
            return None
        
        # Construir grilla desde labels
        grid_dict = {}
        for point in intersections:
            key = tuple(point)
            if key in intersection_labels:
                i, j = intersection_labels[key]
                grid_dict[(i, j)] = point
        
        if len(grid_dict) == 0:
            # Si no hay labels, usar posición espacial
            return self._organize_grid_spatial(intersections)
        
        # Encontrar dimensiones de la grilla
        rows = [coord[0] for coord in grid_dict.keys()]
        cols = [coord[1] for coord in grid_dict.keys()]
        
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        grid_rows = max_row - min_row + 1
        grid_cols = max_col - min_col + 1
        
        # Crear matriz de grilla
        grid_matrix = {}
        points_list = []
        
        for (i, j), point in grid_dict.items():
            normalized_i = i - min_row
            normalized_j = j - min_col
            grid_matrix[(normalized_i, normalized_j)] = point
            points_list.append(point)
        
        return {
            'points': np.array(points_list),
            'grid_matrix': grid_matrix,
            'rows': grid_rows,
            'cols': grid_cols,
            'min_coords': (min_row, min_col)
        }
    
    def _organize_grid_spatial(self, intersections: np.ndarray) -> dict:
        """
        Organiza puntos en grilla usando solo posición espacial.
        Fallback cuando no hay labels disponibles.
        """
        # Agrupar por filas (Y similar)
        sorted_by_y = intersections[np.argsort(intersections[:, 1])]
        
        rows = []
        current_row = [sorted_by_y[0]]
        y_threshold = 30
        
        for point in sorted_by_y[1:]:
            if abs(point[1] - current_row[0][1]) < y_threshold:
                current_row.append(point)
            else:
                current_row.sort(key=lambda p: p[0])
                rows.append(current_row)
                current_row = [point]
        
        if current_row:
            current_row.sort(key=lambda p: p[0])
            rows.append(current_row)
        
        # Crear estructura de grilla
        grid_matrix = {}
        points_list = []
        
        for i, row in enumerate(rows):
            for j, point in enumerate(row):
                grid_matrix[(i, j)] = point
                points_list.append(point)
        
        return {
            'points': np.array(points_list),
            'grid_matrix': grid_matrix,
            'rows': len(rows),
            'cols': max(len(row) for row in rows) if rows else 0,
            'min_coords': (0, 0)
        }
    
    def _create_destination_grid(
        self,
        grid_structure: dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int]]:
        """
        Crea grilla de puntos destino perfectamente alineados.
        
        Args:
            grid_structure: Estructura de grilla organizada
            
        Returns:
            Tupla (src_points, dst_points, board_size)
        """
        grid_matrix = grid_structure['grid_matrix']
        rows = grid_structure['rows']
        cols = grid_structure['cols']
        
        if rows < 2 or cols < 2:
            return None, None, (800, 800)
        
        # Calcular tamaño de celda basado en la mediana de distancias
        cell_size = self._estimate_cell_size(grid_matrix, rows, cols)
        
        # Tamaño del tablero con margen
        margin = 50
        board_width = int(cols * cell_size + 2 * margin)
        board_height = int(rows * cell_size + 2 * margin)
        
        # Crear puntos fuente y destino
        src_points_list = []
        dst_points_list = []
        
        for (i, j), point in grid_matrix.items():
            # Punto fuente
            src_points_list.append(point)
            
            # Punto destino (grilla perfecta)
            dst_x = margin + j * cell_size
            dst_y = margin + i * cell_size
            dst_points_list.append([dst_x, dst_y])
        
        src_points = np.array(src_points_list, dtype=np.float32)
        dst_points = np.array(dst_points_list, dtype=np.float32)
        
        return src_points, dst_points, (board_width, board_height)
    
    def _estimate_cell_size(
        self,
        grid_matrix: dict,
        rows: int,
        cols: int
    ) -> float:
        """
        Estima el tamaño de celda basado en distancias entre puntos adyacentes.
        """
        distances = []
        
        for (i, j), point in grid_matrix.items():
            # Verificar vecino derecho
            if (i, j + 1) in grid_matrix:
                neighbor = grid_matrix[(i, j + 1)]
                dist = np.linalg.norm(point - neighbor)
                distances.append(dist)
            
            # Verificar vecino abajo
            if (i + 1, j) in grid_matrix:
                neighbor = grid_matrix[(i + 1, j)]
                dist = np.linalg.norm(point - neighbor)
                distances.append(dist)
        
        if len(distances) == 0:
            return 100.0
        
        # Usar mediana para ser robusto a outliers
        return float(np.median(distances))
    
    def _create_visualization(
        self,
        original: np.ndarray,
        warped: np.ndarray,
        intersections: np.ndarray,
        src_points: Optional[np.ndarray],
        dst_points: Optional[np.ndarray],
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Crea visualización comparativa antes/después."""
        if len(original.shape) == 2:
            vis_original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis_original = original.copy()
        
        # Dibujar puntos fuente
        if src_points is not None:
            for i, point in enumerate(src_points):
                x, y = map(int, point)
                # Color según si es inlier o outlier
                if mask is not None and mask[i]:
                    color = (0, 255, 0)  # Verde = inlier
                    radius = 5
                else:
                    color = (0, 0, 255)  # Rojo = outlier
                    radius = 3
                cv2.circle(vis_original, (x, y), radius, color, -1)
        
        # Dibujar grilla en imagen warped
        if len(warped.shape) == 2:
            vis_warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        else:
            vis_warped = warped.copy()
        
        if dst_points is not None:
            for point in dst_points:
                x, y = map(int, point)
                if 0 <= x < vis_warped.shape[1] and 0 <= y < vis_warped.shape[0]:
                    cv2.circle(vis_warped, (x, y), 5, (0, 255, 0), -1)
        
        # Redimensionar para comparación
        max_height = 600
        if vis_original.shape[0] > max_height:
            scale = max_height / vis_original.shape[0]
            new_width = int(vis_original.shape[1] * scale)
            vis_original = cv2.resize(vis_original, (new_width, max_height))
        
        if vis_warped.shape[0] > max_height:
            scale = max_height / vis_warped.shape[0]
            new_width = int(vis_warped.shape[1] * scale)
            vis_warped = cv2.resize(vis_warped, (new_width, max_height))
        
        # Concatenar horizontalmente
        # Igualar alturas
        max_h = max(vis_original.shape[0], vis_warped.shape[0])
        if vis_original.shape[0] < max_h:
            padding = max_h - vis_original.shape[0]
            vis_original = cv2.copyMakeBorder(
                vis_original, 0, padding, 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        if vis_warped.shape[0] < max_h:
            padding = max_h - vis_warped.shape[0]
            vis_warped = cv2.copyMakeBorder(
                vis_warped, 0, padding, 0, 0,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        
        comparison = np.hstack([vis_original, vis_warped])
        
        # Etiquetas
        cv2.putText(
            comparison,
            "Original + Grid Points",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        mid = vis_original.shape[1] + 10
        cv2.putText(
            comparison,
            "Warped (Perspective Corrected)",
            (mid, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        # Estadísticas
        inliers = int(np.sum(mask)) if mask is not None else 0
        total = len(src_points) if src_points is not None else 0
        cv2.putText(
            comparison,
            f"Inliers: {inliers}/{total}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return comparison