#!/usr/bin/env python3
"""
Detector de Meeples usando SOLO OpenCV - VERSIÓN MEJORADA
Detecta círculos (meeples) en losetas de Carcassonne
CORRIGE: Falsos positivos con sombras en detección de meeples negros
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class MeepleDetector:
    """Detector de meeples - versión mejorada que NO confunde sombras"""
    
    def __init__(self):
        pass
    
    def detect_meeple_by_color(self, image: np.ndarray) -> Dict:
        """
        Detecta meeples buscando blobs de color azul o negro
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        best_detection = None
        best_score = 0
        
        # DETECTAR AZUL - Rango expandido para capturar más tonos
        # Rango ampliado: [70-150] para H, saturación desde 25
        lower_blue = np.array([70, 25, 30])   # Más permisivo
        upper_blue = np.array([150, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Aplicar morfología para cerrar pequeños huecos
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        blue_result = self._analyze_mask(blue_mask, image, 'blue', w, h)
        
        if blue_result and blue_result['score'] > best_score:
            best_score = blue_result['score']
            best_detection = blue_result
        
        # DETECTAR NEGRO - MÉTODO MEJORADO PARA EVITAR SOMBRAS
        black_result = self._detect_black_meeple_improved(image, hsv, w, h)
        
        if black_result and black_result['score'] > best_score:
            best_score = black_result['score']
            best_detection = black_result
        
        return best_detection
    
    def _detect_black_meeple_improved(self, image: np.ndarray, hsv: np.ndarray, 
                                     w: int, h: int) -> Optional[Dict]:
        """
        Detecta meeples negros con criterios estrictos para evitar sombras
        
        Diferencias clave entre meeple negro y sombra:
        1. Meeple: Color negro PURO (baja saturación, baja luminosidad)
        2. Sombra: Color grisáceo oscuro (saturación variable, no tan oscuro)
        3. Meeple: Forma circular compacta
        4. Sombra: Forma irregular, difusa
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # MÉTODO 1: Threshold adaptativo para manejar variaciones de iluminación
        # Usar threshold más estricto pero con morfología para mantener forma
        _, black_mask_gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # MÉTODO 2: Usar HSV para detectar negro puro
        # Negro puro tiene S baja y V muy baja
        lower_black_hsv = np.array([0, 0, 0])      
        upper_black_hsv = np.array([180, 100, 60])  # Rango un poco más permisivo
        black_mask_hsv = cv2.inRange(hsv, lower_black_hsv, upper_black_hsv)
        
        # COMBINAR ambas máscaras (OR lógico en lugar de AND para no ser tan restrictivo)
        black_mask = cv2.bitwise_or(black_mask_gray, black_mask_hsv)
        
        # Limpiar ruido
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)
        
        # Eliminar ruido pequeño
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # Cerrar huecos pequeños
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        # Analizar la máscara con criterios ESTRICTOS
        return self._analyze_mask_strict(black_mask, image, 'black', w, h)
    
    def _analyze_mask_strict(self, mask: np.ndarray, image: np.ndarray, 
                            color_name: str, w: int, h: int) -> Optional[Dict]:
        """
        Analiza una máscara con CRITERIOS MÁS ESTRICTOS para evitar falsos positivos
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # CRITERIO 1: Área mínima ajustada (4% en lugar de 3%)
            min_area = (w * h) * 0.04
            max_area = (w * h) * 0.55  # Máximo 55%
            
            if area < min_area or area > max_area:
                continue
            
            # CRITERIO 2: Circularidad - más permisivo pero aún estricto
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Debe ser circular (> 0.4 - balance entre estricto y funcional)
            if circularity < 0.4:
                continue
            
            # CRITERIO 3: Verificar COMPACIDAD del blob
            x, y, bw, bh = cv2.boundingRect(contour)
            if bh == 0:
                continue
                
            aspect_ratio = float(bw) / bh
            
            # Debe ser aproximadamente cuadrado - más permisivo
            if aspect_ratio < 0.5 or aspect_ratio > 1.6:
                continue
            
            # CRITERIO 4: Verificar que el blob esté razonablemente CENTRADO
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            bbox_center_x = x + bw // 2
            bbox_center_y = y + bh // 2
            
            center_offset = np.sqrt((cx - bbox_center_x)**2 + (cy - bbox_center_y)**2)
            max_offset = min(bw, bh) * 0.3  # Un poco más permisivo
            
            if center_offset > max_offset:
                continue
            
            # CRITERIO 5: Verificar intensidad promedio
            mask_contour = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour], -1, 255, -1)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Verificar que hay píxeles dentro del contorno
            pixels_in_contour = gray[mask_contour > 0]
            if len(pixels_in_contour) == 0:
                continue
                
            mean_intensity = np.mean(pixels_in_contour)
            
            # Para negro, intensidad promedio debe ser baja (más permisivo: < 70)
            if color_name == 'black' and mean_intensity > 70:
                continue
            
            # CRITERIO 6: Verificar uniformidad del color
            std_intensity = np.std(pixels_in_contour)
            
            # Desviación estándar - más permisivo para meeples reales
            if std_intensity > 30:
                continue
            
            # Calcular score con PESOS AJUSTADOS
            score = (area * 0.4) * (circularity * 0.6)
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None
        
        # Calcular propiedades del mejor contorno
        area = cv2.contourArea(best_contour)
        M = cv2.moments(best_contour)
        
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        radius = int(np.sqrt(area / np.pi))
        
        perimeter = cv2.arcLength(best_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        return {
            'color': color_name,
            'center': (cx, cy),
            'radius': radius,
            'area': area,
            'circularity': circularity,
            'contour': best_contour,
            'score': best_score
        }
    
    def _analyze_mask(self, mask: np.ndarray, image: np.ndarray, 
                     color_name: str, w: int, h: int) -> Optional[Dict]:
        """Analiza una máscara de color y encuentra el mejor blob (versión para azul - más permisiva)"""
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Área mínima REDUCIDA para detectar meeples pequeños (1.5% en lugar de 3%)
            min_area = (w * h) * 0.015  # 1.5%
            max_area = (w * h) * 0.65   # 65%
            
            if area < min_area or area > max_area:
                continue
            
            # Calcular circularidad
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Debe ser razonablemente circular (MÁS PERMISIVO para azul)
            if circularity < 0.2:  # Reducido de 0.25 a 0.2
                continue
            
            # Calcular score - priorizar área para meeples pequeños
            score = area * 1.5 * circularity
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None
        
        # Calcular propiedades del mejor contorno
        area = cv2.contourArea(best_contour)
        M = cv2.moments(best_contour)
        
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        radius = int(np.sqrt(area / np.pi))
        
        perimeter = cv2.arcLength(best_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        return {
            'color': color_name,
            'center': (cx, cy),
            'radius': radius,
            'area': area,
            'circularity': circularity,
            'contour': best_contour,
            'score': best_score
        }
    
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
        print("Uso: python MeepleDetectorSimple.py <imagen>")
        print("Ejemplo: python MeepleDetectorSimple.py foto_meeple.jpg")
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