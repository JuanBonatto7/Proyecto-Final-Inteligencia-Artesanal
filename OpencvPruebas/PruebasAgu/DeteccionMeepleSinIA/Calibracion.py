#!/usr/bin/env python3
"""
Herramienta interactiva para calibrar parámetros del detector
"""

import cv2
import numpy as np
from MeepleDetectorSimple import MeepleDetector


class DetectorCalibrator:
    """Calibrador interactivo de parámetros"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        if self.image is None:
            raise ValueError(f"No se pudo cargar: {image_path}")
        
        self.detector = MeepleDetector()
        self.window_name = "Calibración - Detector de Meeples"
        
        # Parámetros ajustables
        self.params = {
            'dp': 12,           # * 10 para trackbar (1.2 -> 12)
            'minDist': 50,
            'param1': 50,
            'param2': 25,
            'minRadius': 15,
            'maxRadius': 80
        }
    
    def update_detection(self, _):
        """Callback para actualizar detección cuando cambian los parámetros"""
        # Leer valores de trackbars
        self.params['dp'] = cv2.getTrackbarPos('dp x10', self.window_name) / 10.0
        self.params['minDist'] = cv2.getTrackbarPos('minDist', self.window_name)
        self.params['param1'] = cv2.getTrackbarPos('param1', self.window_name)
        self.params['param2'] = cv2.getTrackbarPos('param2', self.window_name)
        self.params['minRadius'] = cv2.getTrackbarPos('minRadius', self.window_name)
        self.params['maxRadius'] = cv2.getTrackbarPos('maxRadius', self.window_name)
        
        # Actualizar detector
        self.detector.circle_params = self.params.copy()
        self.detector.circle_params['dp'] = self.params['dp']
        
        # Detectar círculos
        circles = self.detector.detect_circles(self.image)
        
        # Visualizar
        display = self.image.copy()
        
        # Dibujar grid
        h, w = display.shape[:2]
        cell_w = w // 3
        cell_h = h // 3
        
        for i in range(1, 3):
            cv2.line(display, (cell_w * i, 0), (cell_w * i, h), (200, 200, 200), 1)
            cv2.line(display, (0, cell_h * i), (w, cell_h * i), (200, 200, 200), 1)
        
        # Dibujar círculos detectados
        for i, (x, y, r) in enumerate(circles):
            # Color según el índice (para distinguirlos)
            color = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)][i % 4]
            
            cv2.circle(display, (x, y), r, color, 2)
            cv2.circle(display, (x, y), 2, (0, 0, 255), -1)
            
            # Clasificar color
            meeple_color = self.detector.classify_color(self.image, (x, y, r))
            position = self.detector.get_grid_position((x, y, r), (h, w))
            
            # Etiqueta
            label = f"#{i+1}: {meeple_color} - Pos:{position}"
            cv2.putText(display, label, (x - r, y - r - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info general
        info_text = f"Circulos detectados: {len(circles)}"
        cv2.putText(display, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar parámetros actuales
        y_offset = 60
        param_texts = [
            f"dp: {self.params['dp']:.1f}",
            f"minDist: {self.params['minDist']}",
            f"param1: {self.params['param1']}",
            f"param2: {self.params['param2']}",
            f"minR: {self.params['minRadius']}",
            f"maxR: {self.params['maxRadius']}"
        ]
        
        for text in param_texts:
            cv2.putText(display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        cv2.imshow(self.window_name, display)
    
    def run(self):
        """Ejecutar calibrador interactivo"""
        # Crear ventana
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 800)
        
        # Crear trackbars
        cv2.createTrackbar('dp x10', self.window_name, 
                          int(self.params['dp'] * 10), 30, self.update_detection)
        cv2.createTrackbar('minDist', self.window_name, 
                          self.params['minDist'], 200, self.update_detection)
        cv2.createTrackbar('param1', self.window_name, 
                          self.params['param1'], 200, self.update_detection)
        cv2.createTrackbar('param2', self.window_name, 
                          self.params['param2'], 100, self.update_detection)
        cv2.createTrackbar('minRadius', self.window_name, 
                          self.params['minRadius'], 100, self.update_detection)
        cv2.createTrackbar('maxRadius', self.window_name, 
                          self.params['maxRadius'], 200, self.update_detection)
        
        print("🔧 CALIBRADOR DE DETECTOR DE MEEPLES")
        print("=" * 60)
        print("Ajusta los parámetros usando las barras deslizantes")
        print()
        print("Parámetros:")
        print("  - dp: Resolución del acumulador (1.0-3.0)")
        print("  - minDist: Distancia mínima entre círculos")
        print("  - param1: Umbral Canny (mayor = menos círculos)")
        print("  - param2: Umbral acumulador (menor = más sensible)")
        print("  - minRadius: Radio mínimo del meeple")
        print("  - maxRadius: Radio máximo del meeple")
        print()
        print("Presiona 'S' para guardar parámetros")
        print("Presiona 'Q' o ESC para salir")
        print("=" * 60)
        
        # Dibujar inicial
        self.update_detection(0)
        
        # Loop principal
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q') or key == 27:  # Q o ESC
                print("\n❌ Calibración cancelada")
                break
            
            elif key == ord('s') or key == ord('S'):  # S para guardar
                self.save_params()
                break
        
        cv2.destroyAllWindows()
    
    def save_params(self):
        """Guardar parámetros calibrados"""
        import json
        
        params_to_save = self.params.copy()
        
        output_file = "parametros_calibrados.json"
        with open(output_file, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        
        print("\n✅ Parámetros guardados en:", output_file)
        print("\nParámetros finales:")
        for key, value in params_to_save.items():
            print(f"  {key}: {value}")
        
        print("\n💡 Para usar estos parámetros, copia estos valores en MeepleDetector.circle_params")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python calibrate_detector.py <imagen>")
        print("Ejemplo: python calibrate_detector.py foto_meeple.jpg")
        return
    
    image_path = sys.argv[1]
    
    try:
        calibrator = DetectorCalibrator(image_path)
        calibrator.run()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()