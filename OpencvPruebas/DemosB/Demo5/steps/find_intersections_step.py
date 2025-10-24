"""
Paso 6: Encontrar intersecciones de líneas.
Basado en el proyecto original de Carcassonne tracking.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple
from pipeline.pipeline_step import PipelineStep
from config import Config
from sklearn.cluster import MeanShift, KMeans
from itertools import product


def polar_line_from_segment(segment: np.ndarray) -> np.ndarray:
    """Convierte segmento (x1,y1,x2,y2) a línea polar (rho, theta)"""
    x1, y1, x2, y2 = segment
    
    # Calcular ángulo
    theta = np.arctan2(y2 - y1, x2 - x1)
    
    # Calcular rho (distancia desde origen)
    # Usando la fórmula: rho = x*cos(theta) + y*sin(theta)
    rho = x1 * np.cos(theta) + y1 * np.sin(theta)
    
    # Normalizar para que rho sea siempre positivo
    if rho < 0:
        rho = -rho
        theta = theta + np.pi
    
    # Normalizar theta a [-pi/2, pi/2]
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    
    return np.array([rho, theta])


def intersection_of_polar_lines(img_shape: tuple, line1: np.ndarray, line2: np.ndarray) -> tuple:
    """Calcula intersección entre dos líneas polares"""
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Convertir a forma Ax + By = C
    A1, B1, C1 = np.cos(theta1), np.sin(theta1), rho1
    A2, B2, C2 = np.cos(theta2), np.sin(theta2), rho2
    
    # Resolver sistema
    det = A1 * B2 - A2 * B1
    
    if abs(det) < 1e-10:
        return None  # Líneas paralelas
    
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    
    # Verificar que esté dentro de la imagen
    h, w = img_shape[:2]
    if 0 <= x < w and 0 <= y < h:
        return (int(x), int(y))
    
    return None


class FindIntersectionsStep(PipelineStep):
    """
    Encuentra intersecciones usando el método del proyecto original:
    1. Clasificar líneas en vertical/horizontal (KMeans)
    2. Filtrar outliers (RANSAC simple)
    3. Agrupar líneas paralelas por posición (MeanShift)
    4. Calcular intersecciones con votación ponderada
    5. Filtrar con Non-Maximum Suppression
    """
    
    def __init__(self):
        super().__init__("Find Intersections")
        self.ransac_iterations = 50
        self.ransac_threshold_deg = 10.0
        self.meanshift_bandwidth = 20
        self.nms_window_size = 25
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Encuentra intersecciones entre líneas"""
        
        line_segments = inputs.get('line_segments')
        
        if line_segments is None or len(line_segments) <= 1:
            print('⚠ No hay suficientes líneas!')
            inputs['intersections'] = np.array([])
            inputs['intersection_labels'] = {}
            inputs['debug_image'] = inputs['img'].copy()
            return inputs
        
        img = inputs['img']
        
        print(f"  Procesando {len(line_segments)} líneas...")
        
        # 1. Convertir a polar
        polar_lines = np.array([polar_line_from_segment(s) for s in line_segments])
        
        # 2. Clasificar en vertical/horizontal
        vert_horiz_labels = self._label_vert_horiz(polar_lines)
        
        indices1 = np.where(vert_horiz_labels == 0)[0]
        indices2 = np.where(vert_horiz_labels == 1)[0]
        
        if len(indices1) < 2 or len(indices2) < 2:
            print('⚠ No hay suficientes líneas en ambas direcciones!')
            inputs['intersections'] = np.array([])
            inputs['intersection_labels'] = {}
            inputs['debug_image'] = inputs['img'].copy()
            return inputs
        
        line_segments1 = line_segments[indices1]
        line_segments2 = line_segments[indices2]
        polar_lines1 = polar_lines[indices1]
        polar_lines2 = polar_lines[indices2]
        
        print(f"  Grupo 1: {len(line_segments1)}, Grupo 2: {len(line_segments2)}")
        
        # 3. Filtrar outliers
        polar_lines1, line_segments1 = self._filter_outliers(polar_lines1, line_segments1)
        polar_lines2, line_segments2 = self._filter_outliers(polar_lines2, line_segments2)
        
        print(f"  Después de filtrar: {len(line_segments1)} y {len(line_segments2)}")
        
        # 4. Agrupar líneas paralelas
        labels1 = self._group_parallel_lines(img, polar_lines1)
        labels2 = self._group_parallel_lines(img, polar_lines2)
        
        n_groups1 = len(np.unique(labels1))
        n_groups2 = len(np.unique(labels2))
        print(f"  Grupos: {n_groups1} x {n_groups2}")
        
        # 5. Calcular intersecciones con votación
        intersection_bins, intersection_labels = self._find_intersections_voting(
            img, polar_lines1, line_segments1, labels1,
            polar_lines2, line_segments2, labels2
        )
        
        # 6. Filtrar con NMS
        filtered_intersections = self._apply_nms(intersection_bins)
        
        print(f"  ✅ {len(filtered_intersections)} intersecciones detectadas")
        
        inputs['intersections'] = filtered_intersections
        inputs['intersection_labels'] = intersection_labels
        
        # Visualización
        debug_image = self._create_visualization(
            img, line_segments1, line_segments2,
            labels1, labels2, filtered_intersections, intersection_labels
        )
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _label_vert_horiz(self, polar_lines: np.ndarray) -> np.ndarray:
        """Clasifica líneas en 2 grupos perpendiculares con KMeans"""
        thetas = polar_lines[:, 1]
        thetas_deg = np.rad2deg(thetas)
        
        # KMeans con 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(thetas_deg.reshape(-1, 1))
        
        # Asegurar que grupo 0 tenga ángulos más cercanos a 0° (verticales)
        centers = kmeans.cluster_centers_.flatten()
        if abs(centers[0]) > abs(centers[1]):
            labels = 1 - labels
        
        return labels
    
    def _filter_outliers(
        self,
        polar_lines: np.ndarray,
        line_segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filtrar outliers con RANSAC simple"""
        
        thetas = polar_lines[:, 1]
        thetas_deg = np.rad2deg(thetas)
        
        best_inliers = None
        best_count = 0
        
        for _ in range(self.ransac_iterations):
            # Muestra aleatoria
            sample_idx = np.random.randint(0, len(polar_lines))
            sample_theta = thetas_deg[sample_idx]
            
            # Encontrar inliers (ángulos similares)
            angle_diffs = np.abs(thetas_deg - sample_theta)
            angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
            inlier_mask = angle_diffs < self.ransac_threshold_deg
            
            if np.sum(inlier_mask) > best_count:
                best_count = np.sum(inlier_mask)
                best_inliers = inlier_mask
        
        if best_inliers is None:
            return polar_lines, line_segments
        
        return polar_lines[best_inliers], line_segments[best_inliers]
    
    def _group_parallel_lines(self, img: np.ndarray, polar_lines: np.ndarray) -> np.ndarray:
        """Agrupa líneas paralelas por posición usando MeanShift"""
        
        # Calcular ángulo promedio
        avg_theta = np.mean(polar_lines[:, 1])
        
        # Crear línea perpendicular pasando por el centro
        h, w = img.shape[:2]
        center = np.array([w / 2, h / 2])
        perp_theta = avg_theta + np.pi / 2
        
        # Proyectar cada línea sobre la perpendicular
        # Esto nos da la "posición" de cada línea
        positions = []
        for rho, theta in polar_lines:
            # Punto más cercano al origen en la línea
            x0 = rho * np.cos(theta)
            y0 = rho * np.sin(theta)
            
            # Proyectar sobre perpendicular
            # Simplificación: usar rho directamente como proxy de posición
            positions.append([rho])
        
        positions = np.array(positions)
        
        # MeanShift clustering
        ms = MeanShift(bandwidth=self.meanshift_bandwidth, cluster_all=False)
        labels = ms.fit_predict(positions)
        
        # Manejar outliers (label -1)
        if -1 in labels:
            max_label = np.max(labels)
            labels[labels == -1] = max_label + 1
        
        # 🔧 ORDENAR ESPACIALMENTE
        # Re-etiquetar según posición promedio de cada cluster
        unique_labels = np.unique(labels)
        cluster_positions = []
        
        for label in unique_labels:
            mask = labels == label
            avg_pos = np.mean(positions[mask])
            cluster_positions.append((label, avg_pos))
        
        # Ordenar por posición
        cluster_positions.sort(key=lambda x: x[1])
        
        # Crear mapeo de labels viejos a nuevos (ordenados)
        label_mapping = {old: new for new, (old, _) in enumerate(cluster_positions)}
        
        # Aplicar mapeo
        labels = np.array([label_mapping[l] for l in labels])
        
        return labels
    
    def _find_intersections_voting(
        self,
        img: np.ndarray,
        polar_lines1: np.ndarray,
        line_segments1: np.ndarray,
        labels1: np.ndarray,
        polar_lines2: np.ndarray,
        line_segments2: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """Encuentra intersecciones con sistema de votación ponderada"""
        
        h, w = img.shape[:2]
        intersection_bins = np.zeros((h, w), dtype=np.float32)
        intersection_labels = {}
        
        # Calcular todas las intersecciones
        for (pl1, ls1, i), (pl2, ls2, j) in product(
            zip(polar_lines1, line_segments1, labels1),
            zip(polar_lines2, line_segments2, labels2)
        ):
            point = intersection_of_polar_lines(img.shape, pl1, pl2)
            
            if point is None:
                continue
            
            x, y = point
            
            # Calcular longitudes de segmentos
            len1 = np.linalg.norm(ls1[:2] - ls1[2:])
            len2 = np.linalg.norm(ls2[:2] - ls2[2:])
            
            # Calcular distancias del punto a los extremos
            dist1 = min(
                np.linalg.norm(ls1[:2] - np.array([x, y])),
                np.linalg.norm(ls1[2:] - np.array([x, y]))
            )
            dist2 = min(
                np.linalg.norm(ls2[:2] - np.array([x, y])),
                np.linalg.norm(ls2[2:] - np.array([x, y]))
            )
            
            # Evitar división por cero
            dist1 = max(10, dist1)
            dist2 = max(10, dist2)
            
            # Voto ponderado
            vote = (len1 / dist1) + (len2 / dist2)
            
            intersection_bins[y, x] += vote
            
            # Guardar label del mejor voto
            if (x, y) not in intersection_labels or vote > intersection_bins[y, x]:
                intersection_labels[(x, y)] = (int(i), int(j))
        
        return intersection_bins, intersection_labels
    
    def _apply_nms(self, intersection_bins: np.ndarray) -> np.ndarray:
        """Non-Maximum Suppression para eliminar duplicados"""
        
        filtered = np.copy(intersection_bins)
        
        # Para cada punto con votos
        for y, x in np.argwhere(intersection_bins > 0):
            # Definir ventana
            y_min = max(0, y - self.nms_window_size)
            y_max = min(intersection_bins.shape[0], y + self.nms_window_size)
            x_min = max(0, x - self.nms_window_size)
            x_max = min(intersection_bins.shape[1], x + self.nms_window_size)
            
            window = intersection_bins[y_min:y_max, x_min:x_max]
            
            # Si no es máximo local, eliminar
            if intersection_bins[y, x] < np.max(window):
                filtered[y, x] = 0
        
        # Convertir a lista de coordenadas
        coords = np.argwhere(filtered > 0)
        # Invertir para tener (x, y)
        coords = coords[:, [1, 0]]
        
        return coords
    
    def _create_visualization(
        self,
        img: np.ndarray,
        line_segments1: np.ndarray,
        line_segments2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray,
        intersections: np.ndarray,
        intersection_labels: dict
    ) -> np.ndarray:
        """Visualización con líneas coloreadas por grupo e intersecciones"""
        
        if len(img.shape) == 2:
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            result = img.copy()
        
        # Colores
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
            (0, 128, 255), (128, 255, 0)
        ]
        
        # Dibujar líneas grupo 1
        for ls, label in zip(line_segments1, labels1):
            color = colors[int(label) % len(colors)]
            cv2.line(result, tuple(ls[:2]), tuple(ls[2:]), color, 2)
        
        # Dibujar líneas grupo 2 (con colores diferentes)
        for ls, label in zip(line_segments2, labels2):
            color = colors[(int(label) + 5) % len(colors)]
            cv2.line(result, tuple(ls[:2]), tuple(ls[2:]), color, 2)
        
        # Dibujar intersecciones
        for point in intersections:
            x, y = point
            cv2.circle(result, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(result, (x, y), 8, (255, 255, 255), 1)
            
            # Etiquetar con (i,j)
            if (x, y) in intersection_labels:
                i, j = intersection_labels[(x, y)]
                cv2.putText(
                    result, f"({i},{j})", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1
                )
        
        # Info
        cv2.putText(
            result, f"Intersections: {len(intersections)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        
        return result