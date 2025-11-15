#!/usr/bin/env python3
"""
Detector de Meeples usando SOLO OpenCV
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
    """Detector de meeples usando solo OpenCV - sin IA"""
    
    
    def __init__(self):
        # Parámetros de detección de círculos (Hough Circle Transform)
        # Ajustados para detectar SOLO los meeples grandes (círculos prominentes)
        self.circle_params = {
            'dp': 1.2,              # Resolución del acumulador
            'minDist': 150,         # Distancia mínima entre círculos (MUY aumentada)
            'param1': 80,           # Umbral Canny 
            'param2': 35,           # Umbral acumulador (más alto = más restrictivo)
            'minRadius': 40,        # Radio mínimo GRANDE para meeples
            'maxRadius': 150        # Radio máximo del meeple
        }
        
        
        # Rangos HSV para clasificación de colores
        # Basado en análisis real: Azul HSV(212°,64%,62%), Negro HSV(240°,10%,8%)
        self.color_ranges = {
            'blue': {
                'lower': np.array([90, 100, 80]),    # H: 90-120, S: 100-255, V: 80-255
                'upper': np.array([120, 255, 255])
            },
            'black': {
                'lower': np.array([0, 0, 0]),        # H: cualquiera, S: 0-80, V: 0-70
                'upper': np.array([179, 80, 70])
            }
        }
    
    def detect_circles(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detecta círculos en la imagen usando Hough Circle Transform
        
        Args:
            image: Imagen BGR
            
        Returns:
            Lista de tuplas (x, y, radio) de círculos detectados
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detectar círculos
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.circle_params['dp'],
            minDist=self.circle_params['minDist'],
            param1=self.circle_params['param1'],
            param2=self.circle_params['param2'],
            minRadius=self.circle_params['minRadius'],
            maxRadius=self.circle_params['maxRadius']
        )
        
        if circles is None:
            return []
        
        # Convertir a formato (x, y, radio)
        circles = np.uint16(np.around(circles[0]))
        return [(int(x), int(y), int(r)) for x, y, r in circles]
    
    def classify_color(self, image: np.ndarray, circle: Tuple[int, int, int]) -> str:
        """
        Clasifica el color del meeple usando análisis HSV
        
        Args:
            image: Imagen BGR
            circle: Tupla (x, y, radio)
            
        Returns:
            'blue', 'black' o 'unknown'
        """
        x, y, r = circle
        h, w = image.shape[:2]
        
        # Verificar que el círculo está dentro de la imagen
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            return 'unknown'
        
        # Crear máscara circular (80% del radio para evitar bordes)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), int(r * 0.7), 255, -1)
        
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calcular valor HSV promedio dentro del círculo
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]
        
        # Crear punto HSV para comparar
        hsv_point = np.array(mean_hsv, dtype=np.uint8).reshape(1, 1, 3)
        
        # Verificar contra cada rango de color
        scores = {}
        for color_name, ranges in self.color_ranges.items():
            # Crear máscara para este rango de color
            color_mask = cv2.inRange(
                hsv_point.reshape(1, 1, 3),
                ranges['lower'],
                ranges['upper']
            )
            scores[color_name] = np.sum(color_mask) > 0
        
        # Análisis adicional por canal V (brillo)
        v_value = mean_hsv[2]  # Canal V (Value/Brightness)
        
        # Negro: V bajo (< 70)
        if v_value < 70:
            return 'black'
        
        # Azul: V medio-alto, verificar rango H
        h_value = mean_hsv[0]
        if 90 <= h_value <= 120 and v_value > 80:
            return 'blue'
        
        # Si no hay match claro, usar scores
        if scores.get('blue', False):
            return 'blue'
        if scores.get('black', False):
            return 'black'
        
        return 'unknown'
    
    def get_grid_position(self, circle: Tuple[int, int, int], image_shape: Tuple[int, int]) -> int:
        """
        Determina la posición del meeple en el grid 3x3
        
        Grid:
        0 1 2
        3 4 5
        6 7 8
        
        Args:
            circle: Tupla (x, y, radio)
            image_shape: (height, width)
            
        Returns:
            Posición 0-8, o -1 si está fuera
        """
        x, y, r = circle
        h, w = image_shape
        
        # Tamaño de cada celda
        cell_w = w / 3
        cell_h = h / 3
        
        # Determinar fila y columna
        col = int(x / cell_w)
        row = int(y / cell_h)
        
        # Validar que está dentro del grid
        if 0 <= col < 3 and 0 <= row < 3:
            return row * 3 + col
        
        return -1
    
    def detect_meeple(self, image_path: str) -> Dict:
        """
        Detecta meeple en una imagen
        
        Args:
            image_path: Ruta a la imagen
            
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
        
        # Detectar círculos
        circles = self.detect_circles(image)
        
        if not circles:
            return {
                'has_meeple': False,
                'color': None,
                'position': None,
                'confidence': 0.0,
                'circle': None,
                'image_size': (w, h)
            }
        
        # Tomar el círculo más grande (asumimos que es el meeple)
        best_circle = max(circles, key=lambda c: c[2])
        
        # Clasificar color
        color = self.classify_color(image, best_circle)
        
        # Obtener posición
        position = self.get_grid_position(best_circle, (h, w))
        
        # Calcular confianza basada en:
        # 1. Si el color es conocido (no 'unknown')
        # 2. Si la posición es válida
        confidence = 0.0
        if color != 'unknown':
            confidence += 0.6
        if position != -1:
            confidence += 0.4
        
        return {
            'has_meeple': True,
            'color': color if color != 'unknown' else None,
            'position': position if position != -1 else None,
            'confidence': confidence,
            'circle': best_circle,
            'image_size': (w, h),
            'total_circles_detected': len(circles)
        }
    
    def visualize_detection(self, image_path: str, output_path: Optional[str] = None):
        """
        Visualiza la detección en la imagen
        
        Args:
            image_path: Ruta a la imagen
            output_path: Ruta para guardar (opcional)
        """
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
            # Líneas verticales
            cv2.line(image, (cell_w * i, 0), (cell_w * i, h), (255, 255, 255), 2)
            # Líneas horizontales
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
            
            # Color del círculo según detección
            if result['color'] == 'blue':
                circle_color = (255, 0, 0)  # Azul en BGR
                label_color = 'AZUL'
            elif result['color'] == 'black':
                circle_color = (50, 50, 50)  # Gris oscuro
                label_color = 'NEGRO'
            else:
                circle_color = (0, 255, 255)  # Amarillo para unknown
                label_color = 'DESCONOCIDO'
            
            # Dibujar círculo detectado
            cv2.circle(image, (x, y), r, circle_color, 3)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Centro
            
            # Etiqueta
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
            
            # Info adicional
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
        
        # Mostrar o guardar
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
        print("\n📁 Archivos en el directorio actual:")
        for f in os.listdir('.'):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"   - {f}")
        return
    
    image_path = sys.argv[1]
    
    # Verificar que el archivo existe
    if not os.path.exists(image_path):
        print(f"❌ Archivo no encontrado: {image_path}")
        print(f"\n📁 Buscando archivos de imagen en el directorio actual...")
        found = False
        for f in os.listdir('.'):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                print(f"   - {f}")
                found = True
        if not found:
            print("   No se encontraron imágenes.")
        return
    
    # Crear detector
    detector = MeepleDetector()
    
    # Detectar
    print(f"🔍 Analizando: {image_path}")
    print("-" * 50)
    
    result = detector.detect_meeple(image_path)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
        return
    
    # Mostrar resultados
    print(f"¿Hay meeple?: {'✅ SÍ' if result['has_meeple'] else '❌ NO'}")
    
    if result['has_meeple']:
        print(f"Color: {result['color'] or '❓ Desconocido'}")
        print(f"Posición: {result['position'] if result['position'] is not None else '❓ Fuera del grid'}")
        print(f"Confianza: {result['confidence']:.2%}")
        
        if result['circle']:
            x, y, r = result['circle']
            print(f"Círculo: centro=({x},{y}), radio={r}")
        
        print(f"Total círculos detectados: {result['total_circles_detected']}")
    
    # Visualizar
    print("\n👁️  Generando visualización...")
    output_path = f"deteccion_{Path(image_path).stem}.jpg"
    detector.visualize_detection(image_path, output_path)
    
    print(f"\n✅ Completado!")


if __name__ == "__main__":
    main()