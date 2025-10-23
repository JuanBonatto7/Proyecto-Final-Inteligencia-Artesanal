"""
Paso 6: Encontrar intersecciones de líneas (VERSIÓN MEJORADA).
Basado en clustering y votación ponderada.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from pipeline.pipeline_step import PipelineStep
from config import Config
from sklearn.cluster import MeanShift, KMeans
from itertools import product
from utils.geometry_utils import (
    polar_line_from_segment,
    polar_line_from_point_theta,
    intersection_of_polar_lines
)
from utils.visualization_utils import (
    draw_2point_line_segments
)

class FindIntersectionsStep(PipelineStep):
    """
    Encuentra intersecciones usando clustering de líneas y votación ponderada.
    """
    
    def __init__(
        self,
        min_distance: float = Config.INTERSECTION_MIN_DISTANCE,
        angle_threshold: float = Config.INTERSECTION_ANGLE_THRESHOLD
    ):
        super().__init__("Find Intersections")
        self.min_distance = min_distance
        self.angle_threshold = angle_threshold
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encuentra intersecciones entre líneas agrupadas.
        
        Args:
            inputs: Debe contener 'line_segments'
            
        Returns:
            Inputs actualizado con:
                - 'intersections': Puntos de intersección filtrados
                - 'intersection_labels': Etiquetas de intersecciones
                - 'intersections_img': Imagen binaria de intersecciones
                - 'debug_image': Visualización
        """
        line_segments = inputs.get('line_segments')
        
        if line_segments is None or len(line_segments) <= 1:
            print('⚠ No hay suficientes líneas para encontrar intersecciones!')
            inputs['intersections'] = np.array([])
            inputs['intersection_labels'] = {}
            inputs['intersections_img'] = np.zeros(inputs['img'].shape[:2], dtype=np.uint8)
            inputs['debug_image'] = inputs['img'].copy()
            return inputs
        
        img = inputs['img']
        
        # Convertir segmentos a líneas polares
        polar_lines = np.array([polar_line_from_segment(s) for s in line_segments])
        
        # Etiquetar líneas como verticales u horizontales
        vert_horiz_labels = self._label_vert_horiz_polar_lines(polar_lines)
        
        # Separar en dos grupos
        indices1 = np.nonzero(vert_horiz_labels == 0)[0]
        indices2 = np.nonzero(vert_horiz_labels == 1)[0]
        
        if len(indices1) == 0 or len(indices2) == 0:
            print('⚠ No hay suficientes líneas ortogonales!')
            inputs['intersections'] = np.array([])
            inputs['intersection_labels'] = {}
            inputs['intersections_img'] = np.zeros(img.shape[:2], dtype=np.uint8)
            inputs['debug_image'] = img.copy()
            return inputs
        
        line_segments1 = line_segments[indices1]
        line_segments2 = line_segments[indices2]
        polar_lines1 = polar_lines[indices1]
        polar_lines2 = polar_lines[indices2]
        
        # Filtrar outliers
        polar_lines1, line_segments1 = self._filter_outliers(polar_lines1, line_segments1)
        polar_lines2, line_segments2 = self._filter_outliers(polar_lines2, line_segments2)
        
        # Agrupar líneas paralelas
        labels1, perp_line1 = self._group_polar_lines(img, polar_lines1, line_segments1)
        labels2, perp_line2 = self._group_polar_lines(img, polar_lines2, line_segments2)
        
        # Encontrar intersecciones con votación
        intersection_bins, intersection_labels, intersection_labels_contrib = \
            self._find_intersections_with_voting(
                img, polar_lines1, line_segments1, labels1,
                polar_lines2, line_segments2, labels2
            )
        
        # Filtrar intersecciones usando máximos locales
        filtered_intersections = self._filter_intersections_local_maxima(
            intersection_bins, window_size=25
        )
        
        # Crear imagen de intersecciones
        intersection_img = np.zeros(intersection_bins.shape, dtype=np.uint8)
        intersection_img[intersection_bins > 0] = 255
        
        inputs['intersections'] = filtered_intersections
        inputs['intersection_labels'] = intersection_labels
        inputs['intersections_img'] = intersection_img
        
        # Crear visualización
        debug_image = self._create_visualization(
            img, line_segments1, line_segments2,
            labels1, labels2, intersection_bins, intersection_labels
        )
        inputs['debug_image'] = debug_image
        
        return inputs
    
    def _label_vert_horiz_polar_lines(self, polar_lines: np.ndarray) -> np.ndarray:
        """
        Clasifica líneas en verticales (0) u horizontales (1) usando K-means.
        """
        thetas = polar_lines[:, 1]
        degrees = np.rad2deg(thetas)
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(degrees.reshape(-1, 1))
        labels = kmeans.labels_
        
        # Asegurar que líneas verticales sean 0 (tienen theta cercano a 0 o 180)
        centers = np.squeeze(kmeans.cluster_centers_)
        if centers[0] > centers[1]:
            # Invertir etiquetas
            labels = 1 - labels
        
        return labels
    
    def _filter_outliers(
        self,
        polar_lines: np.ndarray,
        line_segments: np.ndarray,
        num_iterations: int = 50,
        threshold_degrees: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtra outliers usando RANSAC simple.
        Encuentra el conjunto más grande de líneas con ángulos similares.
        """
        best_polar_lines = None
        best_line_segments = None
        best_count = 0
        
        thetas = polar_lines[:, 1]
        thetas_deg = np.rad2deg(thetas)
        
        for _ in range(num_iterations):
            # Seleccionar muestra aleatoria
            sample_idx = np.random.randint(0, len(polar_lines))
            sample_theta = thetas_deg[sample_idx]
            
            # Encontrar inliers
            angle_diffs = np.abs(thetas_deg - sample_theta)
            angle_diffs = np.minimum(angle_diffs, 180 - angle_diffs)
            inlier_indices = np.where(angle_diffs < threshold_degrees)[0]
            
            if len(inlier_indices) > best_count:
                best_count = len(inlier_indices)
                best_polar_lines = polar_lines[inlier_indices]
                best_line_segments = line_segments[inlier_indices]
        
        return best_polar_lines, best_line_segments
    
    def _group_polar_lines(
        self,
        img: np.ndarray,
        polar_lines: np.ndarray,
        line_segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Agrupa líneas paralelas usando MeanShift.
        """
        thetas = polar_lines[:, 1]
        perp_theta = thetas.mean() + np.deg2rad(90)
        
        center = np.array([img.shape[1], img.shape[0]]) / 2
        perp_line = polar_line_from_point_theta(center, perp_theta)
        
        # Proyectar todas las líneas sobre la línea perpendicular
        points = []
        for pl in polar_lines:
            point = intersection_of_polar_lines(img, pl, perp_line)
            if point is None:
                points.append([0, 0])
            else:
                points.append(point)
        points = np.array(points)
        
        # Clustering con MeanShift
        ms = MeanShift(bandwidth=20, cluster_all=False)
        labels = ms.fit_predict(points)
        labels[labels == -1] = np.max(labels) + 1
        
        # Re-etiquetar en orden espacial
        label_mapping = {}
        next_label = 0
        
        # Determinar si ordenar por X o Y
        if np.std(points[:, 0]) > np.std(points[:, 1]):
            ordering = points[:, 0]
        else:
            ordering = points[:, 1]
        
        for i in np.argsort(ordering):
            label = labels[i]
            if label not in label_mapping:
                label_mapping[label] = next_label
                next_label += 1
            labels[i] = label_mapping[label]
        
        return labels, perp_line
    
    def _find_intersections_with_voting(
        self,
        img: np.ndarray,
        polar_lines1: np.ndarray,
        line_segments1: np.ndarray,
        labels1: np.ndarray,
        polar_lines2: np.ndarray,
        line_segments2: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Encuentra intersecciones usando sistema de votación ponderada.
        """
        intersection_labels = {}
        intersection_labels_contrib = {}
        intersection_bins = np.zeros(img.shape[:2])
        
        # Calcular todas las intersecciones
        for (l1, l1_s, i), (l2, l2_s, j) in product(
            zip(polar_lines1, line_segments1, labels1),
            zip(polar_lines2, line_segments2, labels2)
        ):
            point = intersection_of_polar_lines(img, l1, l2)
            if point is None:
                continue
            
            # Calcular longitud de los segmentos
            l1_len = np.linalg.norm(l1_s[:2] - l1_s[2:])
            l2_len = np.linalg.norm(l2_s[:2] - l2_s[2:])
            
            # Calcular distancia del punto a los extremos de los segmentos
            dist_l1 = min(
                np.linalg.norm(l1_s[:2] - point),
                np.linalg.norm(l1_s[2:] - point)
            )
            dist_l2 = min(
                np.linalg.norm(l2_s[:2] - point),
                np.linalg.norm(l2_s[2:] - point)
            )
            
            # Usar distancia mínima para evitar división por cero
            dist_l1 = max(10, dist_l1)
            dist_l2 = max(10, dist_l2)
            
            # Calcular voto basado en proximidad y longitud
            vote = (1/dist_l1 * l1_len) + (1/dist_l2 * l2_len)
            
            x, y = point
            intersection_bins[y, x] += vote
            
            # Guardar la mejor contribución para cada punto
            key = (x, y)
            if key not in intersection_labels_contrib or vote > intersection_labels_contrib[key]:
                intersection_labels[key] = (i, j)
                intersection_labels_contrib[key] = vote
        
        return intersection_bins, intersection_labels, intersection_labels_contrib
    
    def _filter_intersections_local_maxima(
        self,
        intersection_bins: np.ndarray,
        window_size: int = 25
    ) -> np.ndarray:
        """
        Filtra intersecciones manteniendo solo máximos locales.
        """
        filtered_bins = np.copy(intersection_bins)
        
        for (y, x), count in np.ndenumerate(intersection_bins):
            if count == 0:
                continue
            
            # Definir ventana
            x_range = (max(0, x - window_size), min(x + window_size, filtered_bins.shape[1] - 1))
            y_range = (max(0, y - window_size), min(y + window_size, filtered_bins.shape[0] - 1))
            window = intersection_bins[y_range[0]:y_range[1], x_range[0]:x_range[1]]
            
            # Si no es máximo local, eliminar
            if np.max(window) > count:
                filtered_bins[y, x] = 0
        
        # Obtener coordenadas de puntos no cero
        filtered_intersections = np.transpose(np.nonzero(filtered_bins > 0))
        filtered_intersections = np.flip(filtered_intersections, axis=1)
        
        return filtered_intersections
    
    def _create_visualization(
        self,
        image: np.ndarray,
        line_segments1: np.ndarray,
        line_segments2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray,
        intersection_bins: np.ndarray,
        intersection_labels: dict
    ) -> np.ndarray:
        """Crea visualización con líneas agrupadas e intersecciones."""
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()
        
        # Colores para diferentes grupos
        colors = [
            (255, 97, 0), (0, 0, 255), (255, 0, 246), (255, 195, 0),
            (165, 255, 0), (0, 255, 38), (0, 255, 255), (0, 161, 255),
            (128, 0, 128), (255, 128, 0)
        ]
        
        # Dibujar líneas del grupo 1 (verticales/horizontales)
        for i, (ls, label) in enumerate(zip(line_segments1, labels1)):
            color = colors[int(label) % len(colors)]
            draw_2point_line_segments(result, [ls], color=color, thickness=2)
        
        # Dibujar líneas del grupo 2 (perpendiculares al grupo 1)
        for i, (ls, label) in enumerate(zip(line_segments2, labels2)):
            color = colors[int(label) % len(colors)]
            draw_2point_line_segments(result, [ls], color=color, thickness=2)
        
        # Dibujar intersecciones con intensidad según votos
        if np.max(intersection_bins) > 0:
            for p, count in np.ndenumerate(intersection_bins):
                if count > 0:
                    # Color basado en intensidad de votos
                    intensity = int(np.log(count) / np.log(np.max(intersection_bins)) * 255)
                    cv2.circle(result, (p[1], p[0]), 5, (0, 0, intensity), -1)
                    
                    # Etiquetar con coordenadas de grilla
                    if (p[1], p[0]) in intersection_labels:
                        i, j = intersection_labels[(p[1], p[0])]
                        label_text = f"({i},{j})"
                        cv2.putText(
                            result, label_text, (p[1] + 5, p[0] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
                        )
        
        # Información
        num_intersections = np.count_nonzero(intersection_bins > 0)
        cv2.putText(
            result,
            f"Intersections: {num_intersections} | Lines1: {len(line_segments1)} | Lines2: {len(line_segments2)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )
        
        return result