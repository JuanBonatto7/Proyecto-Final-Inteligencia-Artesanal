#!/usr/bin/env python3
"""
Detector de Meeples usando SOLO OpenCV - VERSIÓN ROBUSTA
Detecta círculos (meeples) en losetas de Carcassonne y determina:
- Presencia de meeple
- Posición en grid 3x3 (0-8)
- Color (azul o negro)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class MeepleDetector:
    """Detector de meeples usando solo OpenCV - versión robusta"""
    
    def __init__(self):
        # Rangos de color en HSV
        self.color_ranges = {
            'blue': {
                'lower': np.array([100, 50, 80]),
                'upper': np.array([130, 255, 255])
            },
            'black': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([179, 255, 50])
            }
        }
    
    def detect_meeple_by_color(self, image: np.ndarray) -> Dict:
        """
        Detecta meeples buscando blobs de color específico (azul o negro)
        
        Args:
            image: Imagen BGR
            
        Returns:
            Diccionario con información de detección
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        best_detection = None
        best_score = 0
        
        # Buscar cada color
        for color_name, ranges in self.color_ranges.items():
            # Crear máscara de color
            mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            
            # Aplicar operaciones morfológicas para limpiar
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Analizar cada contorno
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar por área mínima (5% del área total de la imagen)
                min_area = (w * h) * 0.05
                max_area = (w * h) * 0.40
                
                if area < min_area or area > max_area:
                    continue
                
                # Calcular circularidad
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Filtrar por circularidad (debe ser razonablemente circular)
                if circularity < 0.5:
                    continue
                
                # Calcular centro y radio equivalente
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                radius = int(np.sqrt(area / np.pi))
                
                # Calcular score basado en área y circularidad
                score = area * circularity
                
                if score > best_score:
                    best_score = score
                    best_detection = {
                        'color': color_name,
                        'center': (cx, cy),
                        'radius': radius,
                        'area': area,
                        'circularity': circularity,
                        'contour': contour
                    }
        
        return best_detection
    
    def get_grid_position(self, center: Tuple[int, int], image_shape: Tuple[int, int]) -> int:
        """
        Determina la posición en el grid 3x3
        
        Grid:
        0 1 2
        3 4 5
        6 7 8
        """
        x, y = center
        h, w = image_shape
        
        cell_w = w / 3.0
        cell_h = h / 3.0
        
        col = 0 if x < cell_w else (1 if x < cell_w * 2 else 2)
        row = 0 if y < cell_h else (1 if y < cell_h * 2 else 2)
        
        return row * 3 + col
    
    def detect_meeple(self, image_path: str) -> Dict:
        """
        Detecta meeple en una imagen
        
        Returns:
            Diccionario con:
            - has_meeple: bool
            - color: 'blue', 'black' o None
            - position: 0-8 o None
            - confidence: score de confianza
            - circle: (x, y, radio) o None
        """
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                'error': f'No se pudo cargar la imagen: {image_path}',
                'has_meeple': False,
                'color': None,
                'position': None,
                'confidence': 0.0,
                'circle': None
            }
        
        h, w = image.shape[:2]
        
        # Detectar por color
        detection = self.detect_meeple_by_color(image)
        
        if detection is None:
            return {
                'has_meeple': False,
                'color': None,
                'position': None,
                'confidence': 0.0,
                'circle': None,
                'image_size': (w, h)
            }
        
        # Obtener posición
        position = self.get_grid_position(detection['center'], (h, w))
        
        # Calcular confianza basada en circularidad y área
        confidence = min(1.0, detection['circularity'] * 0.7 + 0.3)
        
        return {
            'has_meeple': True,
            'color': detection['color'],
            'position': position,
            'confidence': confidence,
            'circle': (detection['center'][0], detection['center'][1], detection['radius']),
            'image_size': (w, h),
            'area': detection['area'],
            'circularity': detection['circularity']
        }
    
    def visualize_detection(self, image_path: str, output_path: Optional[str] = None):
        """Visualiza la detección en la imagen"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: No se pudo cargar {image_path}")
            return
        
        h, w = image.shape[:2]
        
        # Detectar
        result = self.detect_meeple(image_path)
        
        # Dibujar grid 3x3
        cell_w = w // 3
        cell_h = h // 3
        
        for i in range(1, 3):
            cv2.line(image, (cell_w * i, 0), (cell_w * i, h), (255, 255, 255), 2)
            cv2.line(image, (0, cell_h * i), (w, cell_h * i), (255, 255, 255), 2)
        
        # Dibujar números de posición
        for pos in range(9):
            row = pos // 3
            col = pos % 3
            center_x = col * cell_w + cell_w // 2
            center_y = row * cell_h + cell_h // 2
            cv2.putText(
                image, 
                str(pos), 
                (center_x - 10, center_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (200, 200, 200), 
                2
            )
        
        # Dibujar detección
        if result['has_meeple'] and result['circle']:
            x, y, r = result['circle']
            
            if result['color'] == 'blue':
                circle_color = (255, 0, 0)
                label_color = 'AZUL'
            elif result['color'] == 'black':
                circle_color = (50, 50, 50)
                label_color = 'NEGRO'
            else:
                circle_color = (0, 255, 255)
                label_color = 'DESCONOCIDO'
            
            cv2.circle(image, (x, y), r, circle_color, 3)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            
            label = f"{label_color} - Pos: {result['position']}"
            cv2.putText(
                image, 
                label, 
                (x - r, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                circle_color, 
                2
            )
            
            info = f"Confianza: {result['confidence']:.2f}"
            cv2.putText(
                image, 
                info, 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Visualización guardada en: {output_path}")
        else:
            cv2.imshow('Detección de Meeple', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """Función de prueba"""
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Uso: python meeple_detector.py <imagen>")
        print("Ejemplo: python meeple_detector.py foto_meeple.jpg")
        print("\n🔍 Archivos en el directorio actual:")
        for f in os.listdir('.'):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"   - {f}")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Archivo no encontrado: {image_path}")
        print(f"\n🔍 Buscando archivos de imagen en el directorio actual...")
        found = False
        for f in os.listdir('.'):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"   - {f}")
                found = True
        if not found:
            print("   No se encontraron imágenes.")
        return
    
    detector = MeepleDetector()
    
    print(f"🔍 Analizando: {image_path}")
    print("-" * 50)
    
    result = detector.detect_meeple(image_path)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
        return
    
    print(f"¿Hay meeple?: {'✅ SÍ' if result['has_meeple'] else '❌ NO'}")
    
    if result['has_meeple']:
        print(f"Color: {result['color'] or '❓ Desconocido'}")
        print(f"Posición: {result['position'] if result['position'] is not None else '❓ Fuera del grid'}")
        print(f"Confianza: {result['confidence']:.2%}")
        
        if result['circle']:
            x, y, r = result['circle']
            print(f"Círculo: centro=({x},{y}), radio={r}")
    
    print("\n👁️  Generando visualización...")
    output_path = f"deteccion_{Path(image_path).stem}.jpg"
    detector.visualize_detection(image_path, output_path)
    
    print(f"\n✅ Completado!")


if __name__ == "__main__":
    main()