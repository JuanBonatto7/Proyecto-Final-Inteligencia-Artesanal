"""
Utilidades para cálculos geométricos y líneas polares.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import math


# ============================================================================
# FUNCIONES EXISTENTES (mantener)
# ============================================================================

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calcula distancia euclidiana entre dos puntos."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(line1: Tuple[float, float, float, float],
                   line2: Tuple[float, float, float, float]) -> float:
    """Calcula el ángulo entre dos líneas en grados."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    angle1 = math.atan2(y2 - y1, x2 - x1)
    angle2 = math.atan2(y4 - y3, x4 - x3)
    
    angle_diff = abs(math.degrees(angle1 - angle2))
    
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    return angle_diff


def find_line_intersection(
    line1: Tuple[float, float, float, float],
    line2: Tuple[float, float, float, float]
) -> Optional[Tuple[float, float]]:
    """Encuentra el punto de intersección entre dos líneas."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def cluster_points(points: List[Tuple[float, float]], 
                  max_distance: float) -> List[Tuple[float, float]]:
    """Agrupa puntos cercanos y retorna sus centroides."""
    if not points:
        return []
    
    points = np.array(points)
    clusters = []
    used = np.zeros(len(points), dtype=bool)
    
    for i, point in enumerate(points):
        if used[i]:
            continue
        
        distances = np.sqrt(np.sum((points - point)**2, axis=1))
        cluster_mask = distances < max_distance
        cluster_points = points[cluster_mask]
        
        used[cluster_mask] = True
        
        centroid = np.mean(cluster_points, axis=0)
        clusters.append(tuple(centroid))
    
    return clusters


def sort_points_grid(points: List[Tuple[float, float]]) -> np.ndarray:
    """Ordena puntos en una grilla de arriba-abajo, izquierda-derecha."""
    if not points:
        return np.array([])
    
    points = np.array(points)
    sorted_indices = np.lexsort((points[:, 0], points[:, 1]))
    
    return points[sorted_indices]


def is_point_inside_polygon(point: Tuple[float, float], 
                            polygon: np.ndarray) -> bool:
    """Verifica si un punto está dentro de un polígono."""
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0


def calculate_homography_ransac(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    threshold: float = 5.0,
    max_iterations: int = 2000
) -> Optional[np.ndarray]:
    """Calcula homografía usando RANSAC."""
    if len(src_points) < 4 or len(dst_points) < 4:
        return None
    
    src_points = src_points.astype(np.float32)
    dst_points = dst_points.astype(np.float32)
    
    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        threshold,
        maxIters=max_iterations
    )
    
    return H


# ============================================================================
# FUNCIONES NUEVAS PARA LÍNEAS POLARES
# ============================================================================

def polar_line_from_segment(segment: np.ndarray) -> np.ndarray:
    """
    Convierte un segmento de línea (x1, y1, x2, y2) a formato polar (rho, theta).
    
    Args:
        segment: Array [x1, y1, x2, y2]
        
    Returns:
        Array [rho, theta]
    """
    x1, y1, x2, y2 = segment
    
    # Calcular el ángulo de la línea
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan2(dy, dx)
    
    # Normalizar theta a [0, pi]
    if theta < 0:
        theta += np.pi
    
    # Calcular rho (distancia perpendicular desde el origen)
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    
    rho = x_mid * np.cos(theta) + y_mid * np.sin(theta)
    
    # Asegurar que rho sea positivo
    if rho < 0:
        rho = -rho
        theta = theta - np.pi if theta >= np.pi else theta + np.pi
    
    return np.array([rho, theta])


def polar_line_from_point_theta(point: np.ndarray, theta: float) -> np.ndarray:
    """
    Crea una línea polar que pasa por un punto con un ángulo dado.
    
    Args:
        point: Punto [x, y]
        theta: Ángulo en radianes
        
    Returns:
        Array [rho, theta]
    """
    x, y = point
    rho = x * np.cos(theta) + y * np.sin(theta)
    
    if rho < 0:
        rho = -rho
        theta = theta - np.pi if theta >= np.pi else theta + np.pi
    
    return np.array([rho, theta])


def intersection_of_polar_lines(
    img: np.ndarray,
    line1: np.ndarray,
    line2: np.ndarray
) -> Optional[np.ndarray]:
    """
    Encuentra la intersección de dos líneas en formato polar.
    
    Args:
        img: Imagen (para obtener dimensiones)
        line1: Primera línea [rho1, theta1]
        line2: Segunda línea [rho2, theta2]
        
    Returns:
        Punto de intersección [x, y] o None si no hay intersección
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Convertir a forma cartesiana: ax + by = c
    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    c1 = rho1
    
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    c2 = rho2
    
    # Resolver el sistema de ecuaciones
    det = a1 * b2 - a2 * b1
    
    if abs(det) < 1e-10:
        return None
    
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    
    # Verificar que el punto esté dentro de la imagen
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        return np.array([int(x), int(y)])
    
    return None


def merge_close_lines(
    polar_lines: np.ndarray,
    rho_threshold: float = 20,
    theta_threshold: float = 0.1
) -> np.ndarray:
    """Fusiona líneas que están muy cerca entre sí."""
    if polar_lines is None or len(polar_lines) == 0:
        return polar_lines
    
    merged = []
    used = np.zeros(len(polar_lines), dtype=bool)
    
    for i in range(len(polar_lines)):
        if used[i]:
            continue
        
        rho_i, theta_i = polar_lines[i]
        group_rhos = [rho_i]
        group_thetas = [theta_i]
        used[i] = True
        
        for j in range(i + 1, len(polar_lines)):
            if used[j]:
                continue
            
            rho_j, theta_j = polar_lines[j]
            
            theta_diff = abs(theta_i - theta_j)
            theta_diff = min(theta_diff, np.pi - theta_diff)
            
            if abs(rho_i - rho_j) < rho_threshold and theta_diff < theta_threshold:
                group_rhos.append(rho_j)
                group_thetas.append(theta_j)
                used[j] = True
        
        avg_rho = np.mean(group_rhos)
        avg_theta = np.mean(group_thetas)
        merged.append([avg_rho, avg_theta])
    
    return np.array(merged)


def filter_lines_by_length(
    line_segments: np.ndarray,
    min_length: float
) -> np.ndarray:
    """Filtra segmentos de línea por longitud mínima."""
    if line_segments is None or len(line_segments) == 0:
        return line_segments
    
    lengths = np.sqrt(
        (line_segments[:, 2] - line_segments[:, 0])**2 +
        (line_segments[:, 3] - line_segments[:, 1])**2
    )
    
    return line_segments[lengths >= min_length]


def cartesian_to_polar_line(a: float, b: float, c: float) -> Tuple[float, float]:
    """Convierte línea de forma cartesiana ax + by = c a formato polar."""
    norm = np.sqrt(a**2 + b**2)
    if norm == 0:
        return 0, 0
    
    a_norm = a / norm
    b_norm = b / norm
    c_norm = c / norm
    
    theta = np.arctan2(b_norm, a_norm)
    rho = c_norm
    
    if rho < 0:
        rho = -rho
        theta = theta + np.pi
    
    theta = theta % np.pi
    
    return rho, theta


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rota una imagen por un ángulo dado."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated