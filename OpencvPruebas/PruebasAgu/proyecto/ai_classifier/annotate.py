"""
Herramienta de Anotación Interactiva para Losetas de Carcassonne

Permite etiquetar las losetas de forma fácil y rápida con:
- Tipo de loseta (A-X + BLANCO)
- Rotación (0-3)
- Presencia de meeple (Sí/No)
- Posición del meeple (0-8)
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


class AnnotationTool:
    """Herramienta interactiva para anotar losetas."""
    
    # Tipos de losetas
    TILE_TYPES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'BLANCO'
    ]
    
    def __init__(self, tiles_dir: str, output_file: str = 'annotations.json'):
        """
        Inicializa la herramienta.
        
        Args:
            tiles_dir: Directorio con las imágenes de losetas
            output_file: Archivo donde guardar las anotaciones
        """
        self.tiles_dir = tiles_dir
        self.output_file = output_file
        
        # Cargar imágenes
        self.image_paths = self._load_image_paths()
        self.current_idx = 0
        
        # Cargar anotaciones existentes
        self.annotations = self._load_annotations()
        
        # Estado actual
        self.current_annotation = {
            'tile_letter': 'A',
            'tile_type': 0,
            'rotation': 0,
            'has_meeple': False,
            'meeple_position': 0,
            'meeple_color': 'blue'  # Por defecto blue
        }
        
        # Control de actualización de UI
        self.needs_redraw = True
        
        # Códigos de teclas calibrados
        self.arrow_left_codes = []
        self.arrow_right_codes = []
        
        print(f"✓ Cargadas {len(self.image_paths)} imágenes")
        print(f"✓ {len(self.annotations)} ya anotadas")
    
    def _load_image_paths(self) -> List[str]:
        """Carga las rutas de todas las imágenes."""
        patterns = ['*.png', '*.jpg', '*.jpeg']
        image_paths = []
        
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(self.tiles_dir, pattern)))
        
        return sorted(image_paths)
    
    def _load_annotations(self) -> Dict[str, Dict]:
        """Carga anotaciones existentes."""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                ann_list = json.load(f)
            
            # Convertir a diccionario por ruta de imagen
            annotations = {}
            for ann in ann_list:
                annotations[ann['image_path']] = ann
            
            return annotations
        return {}
    
    def _save_annotations(self):
        """Guarda las anotaciones en el archivo."""
        ann_list = list(self.annotations.values())
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(ann_list, f, indent=2)
        print(f"✓ Anotaciones guardadas: {len(ann_list)} muestras")
    
    def _get_relative_path(self, full_path: str) -> str:
        """Convierte una ruta absoluta a relativa desde tiles_dir."""
        return os.path.relpath(full_path, start=os.path.dirname(self.tiles_dir))
    
    def _draw_interface(self, image: np.ndarray) -> np.ndarray:
        """Dibuja la interfaz sobre la imagen."""
        h, w = image.shape[:2]
        
        # Panel lateral más amplio para mayor claridad
        panel_width = 300
        # Asegurar altura mínima
        min_height = 650
        display_height = max(h, min_height)
        
        canvas = np.zeros((display_height, w + panel_width, 3), dtype=np.uint8)
        canvas[:h, :w] = image
        
        # Fondo sólido oscuro
        canvas[:, w:] = (40, 40, 40)
        
        # Función para dibujar texto
        def draw_text(text, y, color=(255, 255, 255), size=0.55, bold=False, indent=12):
            thickness = 2 if bold else 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            x_pos = w + indent
            cv2.putText(canvas, text, (x_pos, int(y)), font, size, color, thickness, cv2.LINE_AA)
        
        def draw_box(y, height, color=(60, 60, 60), border_color=(100, 100, 100)):
            y_int = int(y)
            h_int = int(height)
            cv2.rectangle(canvas, (w + 8, y_int), (w + panel_width - 8, y_int + h_int), color, -1)
            cv2.rectangle(canvas, (w + 8, y_int), (w + panel_width - 8, y_int + h_int), border_color, 2)
        
        # Verificar si esta loseta ya está guardada
        current_path = self.image_paths[self.current_idx]
        rel_path = self._get_relative_path(current_path)
        is_saved = rel_path in self.annotations
        
        y_offset = 15
        
        # ============ HEADER ============
        header_height = 55
        header_color = (50, 100, 50) if is_saved else (80, 60, 50)
        header_border = (100, 180, 100) if is_saved else (150, 100, 80)
        draw_box(y_offset, header_height, header_color, header_border)
        
        draw_text("ANOTACION", y_offset + 25, (255, 255, 255), 0.6, bold=True, indent=85)
        progress = f"{self.current_idx + 1}/{len(self.image_paths)}"
        status = "OK" if is_saved else "--"
        status_color = (100, 255, 100) if is_saved else (255, 150, 100)
        draw_text(f"{progress} [{status}]", y_offset + 45, status_color, 0.5, bold=True, indent=95)
        
        y_offset += header_height + 10
        
        # ============ TIPO LOSETA ============
        tile_height = 70
        draw_box(y_offset, tile_height, (45, 60, 45), (80, 120, 80))
        draw_text("TIPO", y_offset + 22, (150, 255, 150), 0.5, bold=True)
        
        letter = self.current_annotation['tile_letter']
        letter_display = letter if len(letter) <= 6 else "BLANCO"
        tile_idx = self.current_annotation['tile_type']
        
        # Letra grande
        cv2.putText(canvas, letter_display, (w + 18, int(y_offset + 55)), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (120, 255, 120), 2, cv2.LINE_AA)
        
        # Info al lado
        draw_text(f"idx:{tile_idx}", y_offset + 40, (180, 220, 180), 0.45, indent=170)
        draw_text("A-X/SPACE", y_offset + 60, (120, 120, 120), 0.38, indent=170)
        
        y_offset += tile_height + 10
        
        # ============ ROTACION ============
        rot_height = 65
        draw_box(y_offset, rot_height, (60, 45, 45), (120, 80, 80))
        draw_text("ROTACION", y_offset + 22, (150, 200, 255), 0.5, bold=True)
        
        rotation = self.current_annotation['rotation']
        degrees = rotation * 90
        arrows = [">", "v", "<", "^"]
        arrow = arrows[rotation]
        
        # Número y flecha
        cv2.putText(canvas, f"{rotation}", (w + 18, int(y_offset + 52)), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.3, (120, 200, 255), 2, cv2.LINE_AA)
        
        cv2.putText(canvas, arrow, (w + 70, int(y_offset + 52)), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.1, (180, 220, 255), 2, cv2.LINE_AA)
        
        # Info
        draw_text(f"{degrees}", y_offset + 42, (180, 220, 255), 0.45, indent=170)
        draw_text("+/-: Rot", y_offset + 60, (120, 120, 120), 0.38, indent=170)
        
        y_offset += rot_height + 10
        
        # ============ MEEPLE ============
        has_meeple = self.current_annotation['has_meeple']
        meeple_bg = (45, 60, 45) if has_meeple else (60, 45, 45)
        meeple_border = (80, 120, 80) if has_meeple else (120, 80, 80)
        
        meeple_height = 110 if has_meeple else 65
        draw_box(y_offset, meeple_height, meeple_bg, meeple_border)
        draw_text("MEEPLE", y_offset + 22, (255, 200, 150), 0.5, bold=True)
        
        meeple_status = "SI" if has_meeple else "NO"
        meeple_color = (120, 255, 120) if has_meeple else (120, 120, 255)
        
        cv2.putText(canvas, meeple_status, (w + 18, int(y_offset + 52)), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.1, meeple_color, 2, cv2.LINE_AA)
        
        draw_text("TAB:toggle", y_offset + 42, (120, 120, 120), 0.38, indent=170)
        
        # Posición y color si tiene meeple
        if has_meeple:
            pos = self.current_annotation['meeple_position']
            color = self.current_annotation['meeple_color']
            
            draw_text(f"POS: {pos}", y_offset + 70, (255, 220, 150), 0.55, indent=18)
            draw_text("0-8", y_offset + 80, (120, 120, 120), 0.38, indent=170)
            
            # Mostrar color con color visual
            color_display = color.upper()
            color_rgb = (255, 180, 100) if color == 'blue' else (180, 180, 180)
            draw_text(f"COLOR: {color_display}", y_offset + 95, color_rgb, 0.55, indent=18)
            draw_text("B/N", y_offset + 105, (120, 120, 120), 0.38, indent=170)
        
        y_offset += meeple_height + 10
        
        # ============ CONTROLES ============
        remaining_space = display_height - y_offset - 15
        ctrl_height = max(remaining_space, 180)
        
        draw_box(y_offset, ctrl_height, (50, 50, 70), (100, 100, 120))
        draw_text("CONTROLES", y_offset + 25, (255, 255, 100), 0.5, bold=True, indent=90)
        
        # Lista de controles con mejor espaciado
        y_c = y_offset + 50
        line_height = 22
        
        draw_text("A-X: Tipo loseta", y_c, (200, 220, 255), 0.42)
        y_c += line_height
        draw_text("SPACE: BLANCO", y_c, (200, 220, 255), 0.42)
        y_c += line_height
        draw_text("+/-: Rotar", y_c, (200, 220, 255), 0.42)
        y_c += line_height
        draw_text("TAB: Meeple on/off", y_c, (200, 220, 255), 0.42)
        y_c += line_height
        draw_text("0-8: Pos meeple", y_c, (200, 220, 255), 0.42)
        y_c += line_height
        draw_text("B/N: Color meeple", y_c, (200, 220, 255), 0.42)
        y_c += line_height + 5
        draw_text("ENTER: Guardar+Sig", y_c, (150, 255, 150), 0.42, bold=True)
        y_c += line_height
        draw_text("< >: Navegar", y_c, (150, 255, 150), 0.42, bold=True)
        y_c += line_height
        draw_text("F5: Guardar", y_c, (255, 255, 150), 0.42, bold=True)
        y_c += line_height
        draw_text("ESC: Salir", y_c, (255, 150, 150), 0.42, bold=True)
        
        return canvas
    
    def _load_current_image(self) -> Optional[np.ndarray]:
        """Carga la imagen actual."""
        if 0 <= self.current_idx < len(self.image_paths):
            path = self.image_paths[self.current_idx]
            image = cv2.imread(path)
            
            if image is not None:
                # Redimensionar para tener un tamaño consistente
                h, w = image.shape[:2]
                target_size = 600  # Más grande para mejor visualización
                if h > target_size or w > target_size:
                    scale = target_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                else:
                    # Si es muy pequeña, agrandarla un poco
                    if h < 400 or w < 400:
                        scale = 400 / min(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        image = cv2.resize(image, (new_w, new_h))
            
            # Cargar anotación existente si hay
            rel_path = self._get_relative_path(path)
            if rel_path in self.annotations:
                ann = self.annotations[rel_path]
                self.current_annotation = {
                    'tile_letter': ann['tile_letter'],
                    'tile_type': ann['tile_type'],
                    'rotation': ann['rotation'],
                    'has_meeple': ann['has_meeple'],
                    'meeple_position': ann['meeple_position'],
                    'meeple_color': ann.get('meeple_color', 'blue')  # Compatibilidad con anotaciones antiguas
                }
            else:
                # Reset a valores por defecto
                self.current_annotation = {
                    'tile_letter': 'A',
                    'tile_type': 0,
                    'rotation': 0,
                    'has_meeple': False,
                    'meeple_position': 0,
                    'meeple_color': 'blue'
                }
            
            self.needs_redraw = True
            return image
        return None
    
    def _calibrate_arrow_keys(self):
        """Calibra las teclas de flecha para este sistema."""
        print("\n" + "="*70)
        print("CALIBRACIÓN DE NAVEGACIÓN")
        print("="*70)
        print("\n💡 IMPORTANTE: Las flechas pueden no funcionar en todos los sistemas.")
        print("   Por eso también puedes usar:")
        print("     < (coma) = imagen anterior")
        print("     > (punto) = imagen siguiente")
        print("\n¿Quieres intentar calibrar las flechas? (S/N)")
        print("Presiona S para saltar, o cualquier otra tecla para calibrar.")
        print("="*70)
        
        # Crear ventana temporal pequeña
        calib_window = "Calibracion"
        canvas = np.zeros((150, 500, 3), dtype=np.uint8)
        canvas[:] = (40, 40, 40)
        
        cv2.putText(canvas, "S = Saltar | Otra = Calibrar", (80, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.namedWindow(calib_window, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(calib_window, canvas)
        
        # Esperar decisión
        decision_key = cv2.waitKey(5000) & 0xFF
        if decision_key == ord('s') or decision_key == ord('S'):
            print("\n✓ Calibración saltada.")
            print("  Usa < para retroceder y > para avanzar")
            cv2.destroyWindow(calib_window)
            return
        
        # Calibrar flecha izquierda
        canvas[:] = (40, 40, 40)
        cv2.putText(canvas, "Presiona FLECHA IZQUIERDA <--", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        cv2.imshow(calib_window, canvas)
        
        print("\n⏳ Esperando flecha IZQUIERDA...")
        left_detected = False
        
        for i in range(50):  # 50 intentos de 200ms = 10 segundos
            key_code = cv2.waitKey(200)
            if key_code != -1 and key_code != 0:
                key = key_code & 0xFF
                self.arrow_left_codes = [key, key_code]
                print(f"✓ IZQUIERDA: key={key}, code={key_code}")
                left_detected = True
                break
        
        if not left_detected:
            print("✗ No detectada. Usa < para retroceder.")
        
        # Calibrar flecha derecha
        canvas[:] = (40, 40, 40)
        cv2.putText(canvas, "Presiona FLECHA DERECHA -->", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.imshow(calib_window, canvas)
        
        print("⏳ Esperando flecha DERECHA...")
        right_detected = False
        
        for i in range(50):  # 50 intentos de 200ms = 10 segundos
            key_code = cv2.waitKey(200)
            if key_code != -1 and key_code != 0:
                key = key_code & 0xFF
                # Solo aceptar si es diferente a la izquierda
                if not left_detected or (key_code != self.arrow_left_codes[1]):
                    self.arrow_right_codes = [key, key_code]
                    print(f"✓ DERECHA: key={key}, code={key_code}")
                    right_detected = True
                    break
        
        if not right_detected:
            print("✗ No detectada. Usa > para avanzar.")
        
        cv2.destroyWindow(calib_window)
        
        if left_detected and right_detected:
            print("\n✅ Calibración exitosa!")
        else:
            print("\n⚠️ Calibración incompleta.")
        print("  Siempre puedes usar: < (atrás) y > (adelante)")
        print("="*70 + "\n")
    
    def _update_display_now(self, window_name: str):
        """Actualiza el display inmediatamente sin esperar al siguiente ciclo."""
        if 0 <= self.current_idx < len(self.image_paths):
            path = self.image_paths[self.current_idx]
            image = cv2.imread(path)
            
            if image is not None:
                # Redimensionar
                h, w = image.shape[:2]
                target_size = 600
                if h > target_size or w > target_size:
                    scale = target_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                else:
                    if h < 400 or w < 400:
                        scale = 400 / min(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        image = cv2.resize(image, (new_w, new_h))
                
                # Dibujar interfaz con las anotaciones ACTUALES (sin recargar desde archivo)
                display = self._draw_interface(image)
                cv2.imshow(window_name, display)
                cv2.waitKey(1)  # Forzar actualización del sistema de ventanas
    
    def _save_current_annotation(self):
        """Guarda la anotación actual."""
        if 0 <= self.current_idx < len(self.image_paths):
            path = self.image_paths[self.current_idx]
            rel_path = self._get_relative_path(path)
            
            # Si no hay meeple, guardar posición como -1 y color como None para compatibilidad
            meeple_pos = self.current_annotation['meeple_position'] if self.current_annotation['has_meeple'] else -1
            meeple_color = self.current_annotation['meeple_color'] if self.current_annotation['has_meeple'] else None
            
            annotation = {
                'image_path': rel_path,
                'tile_letter': self.current_annotation['tile_letter'],
                'tile_type': self.current_annotation['tile_type'],
                'rotation': self.current_annotation['rotation'],
                'has_meeple': self.current_annotation['has_meeple'],
                'meeple_position': meeple_pos,
                'meeple_color': meeple_color,
                'confidence': 1.0,
                'auto_annotated': False
            }
            
            self.annotations[rel_path] = annotation
            color_info = f", color={meeple_color}" if self.current_annotation['has_meeple'] else ""
            print(f"✓ Guardado: {self.current_annotation['tile_letter']}, rot={self.current_annotation['rotation']}, meeple={self.current_annotation['has_meeple']}{color_info}")
    
    def run(self):
        """Ejecuta la herramienta de anotación."""
        print("\n" + "="*70)
        print("HERRAMIENTA DE ANOTACIÓN DE LOSETAS - MEJORADA")
        print("="*70)
        print("\n📋 CONTROLES:")
        print("  🔤 A-X: Seleccionar tipo de loseta")
        print("  ⬜ ESPACIO: Loseta BLANCO")
        print("  🔄 +/-: Rotar derecha/izquierda")
        print("  👤 TAB: Toggle presencia de meeple")
        print("  📍 0-8: Posición del meeple (solo si hay meeple)")
        print("  🎨 B: Color BLUE del meeple (solo si hay meeple)")
        print("  🎨 N: Color BLACK del meeple (solo si hay meeple)")
        print("")
        print("  ✅ ENTER: Guardar y siguiente")
        print("  💾 F5: Guardar progreso sin avanzar")
        print("  ◀▶ < y >: Navegar (< atrás, > adelante) ⭐ RECOMENDADO")
        print("  🚪 ESC: Salir y guardar")
        print("="*70 + "\n")
        
        # Calibrar teclas de flecha
        self._calibrate_arrow_keys()
        
        window_name = "Anotacion de Losetas"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Cargar primera imagen
        image = self._load_current_image()
        if image is None:
            print("No hay imágenes para anotar")
            return
        
        # Crear display inicial y ajustar ventana
        display = self._draw_interface(image)
        h, w = display.shape[:2]
        cv2.resizeWindow(window_name, w, h)
        cv2.imshow(window_name, display)
        self.needs_redraw = False
        
        while True:
            # Esperar tecla con timeout corto para mantener UI responsive
            key_code = cv2.waitKey(1)
            
            # Redibujar si es necesario (hacerlo después de waitKey para ser más responsive)
            if self.needs_redraw:
                image = self._load_current_image()
                if image is None:
                    break
                display = self._draw_interface(image)
                cv2.imshow(window_name, display)
                cv2.waitKey(1)  # Pequeña pausa para forzar actualización
                self.needs_redraw = False
            
            # Si no hay tecla presionada, continuar
            if key_code == -1:
                continue
            
            key = key_code & 0xFF
            
            # Procesamiento de teclas
            handled = False
            
            # ESC - Salir
            if key == 27:
                print("\nSaliendo...")
                break
            
            # ENTER - Guardar y siguiente
            elif key == 13:
                self._save_current_annotation()
                self.current_idx += 1
                if self.current_idx >= len(self.image_paths):
                    print("\n¡Todas las imágenes anotadas!")
                    break
                self._update_display_now(window_name)
                handled = True
            
            # Flechas - navegación con múltiples métodos de detección
            # Flecha derecha - avanzar
            elif (len(self.arrow_right_codes) > 0 and (key in self.arrow_right_codes or key_code in self.arrow_right_codes)) or \
                 key == 83 or key_code == 2555904 or key_code == 65363 or \
                 (key_code >> 16) == 77:  # Código extendido Windows
                self.current_idx = min(self.current_idx + 1, len(self.image_paths) - 1)
                print(f"→ Imagen {self.current_idx + 1}/{len(self.image_paths)}")
                self._update_display_now(window_name)
                handled = True
            
            # Flecha izquierda - retroceder
            elif (len(self.arrow_left_codes) > 0 and (key in self.arrow_left_codes or key_code in self.arrow_left_codes)) or \
                 key == 81 or key_code == 2424832 or key_code == 65361 or \
                 (key_code >> 16) == 75:  # Código extendido Windows
                self.current_idx = max(self.current_idx - 1, 0)
                print(f"← Imagen {self.current_idx + 1}/{len(self.image_paths)}")
                self._update_display_now(window_name)
                handled = True
            
            # < y > - NAVEGACIÓN PRINCIPAL (siempre funciona)
            elif key == ord(',') or key == ord('<') or key == 188 or key == 60:  # < para anterior
                self.current_idx = max(self.current_idx - 1, 0)
                print(f"← Imagen {self.current_idx + 1}/{len(self.image_paths)}")
                self._update_display_now(window_name)
                handled = True
            
            elif key == ord('.') or key == ord('>') or key == 190 or key == 62:  # > para siguiente
                self.current_idx = min(self.current_idx + 1, len(self.image_paths) - 1)
                print(f"→ Imagen {self.current_idx + 1}/{len(self.image_paths)}")
                self._update_display_now(window_name)
                handled = True
            
            # + o = - Rotar derecha (0 -> 1 -> 2 -> 3 -> 0)
            elif key == ord('+') or key == ord('=') or key == 187 or key == 43:
                old_rot = self.current_annotation['rotation']
                self.current_annotation['rotation'] = (old_rot + 1) % 4
                rotation = self.current_annotation['rotation']
                degrees = rotation * 90
                print(f"✓ Rotación: {old_rot} → {rotation} ({degrees}°) [tecla: {key}]")
                self._update_display_now(window_name)
                handled = True
            
            # - o _ - Rotar izquierda (0 -> 3 -> 2 -> 1 -> 0)
            elif key == ord('-') or key == ord('_') or key == 189 or key == 45:
                old_rot = self.current_annotation['rotation']
                self.current_annotation['rotation'] = (old_rot - 1 + 4) % 4
                rotation = self.current_annotation['rotation']
                degrees = rotation * 90
                print(f"✓ Rotación: {old_rot} → {rotation} ({degrees}°) [tecla: {key}]")
                self._update_display_now(window_name)
                handled = True
            
            # TAB - Toggle meeple
            elif key == 9:  # TAB
                self.current_annotation['has_meeple'] = not self.current_annotation['has_meeple']
                if not self.current_annotation['has_meeple']:
                    self.current_annotation['meeple_position'] = 0
                print(f"✓ Meeple: {'SÍ' if self.current_annotation['has_meeple'] else 'NO'}")
                self._update_display_now(window_name)
                handled = True
            
            # 0-8 - Posición del meeple (solo si tiene meeple)
            elif 48 <= key <= 56:  # 0-8
                if self.current_annotation['has_meeple']:
                    position = key - 48
                    self.current_annotation['meeple_position'] = position
                    print(f"✓ Posición meeple: {position}")
                    self._update_display_now(window_name)
                    handled = True
                else:
                    print("⚠ Primero activa el meeple con TAB")
                    handled = True  # Marcar como manejado para no mostrar "no reconocida"
            
            # B - Color blue del meeple (solo si tiene meeple)
            elif key == ord('b') or key == ord('B'):
                if self.current_annotation['has_meeple']:
                    self.current_annotation['meeple_color'] = 'blue'
                    print(f"✓ Color meeple: BLUE")
                    self._update_display_now(window_name)
                    handled = True
                else:
                    print("⚠ Primero activa el meeple con TAB")
                    handled = True
            
            # N - Color black del meeple (solo si tiene meeple)
            elif key == ord('n') or key == ord('N'):
                if self.current_annotation['has_meeple']:
                    self.current_annotation['meeple_color'] = 'black'
                    print(f"✓ Color meeple: BLACK")
                    self._update_display_now(window_name)
                    handled = True
                else:
                    print("⚠ Primero activa el meeple con TAB")
                    handled = True
            
            # F5 - Guardar progreso
            elif key == 196 or key == 63:  # F5
                self._save_current_annotation()
                self._save_annotations()
                print("💾 Progreso guardado")
                self._update_display_now(window_name)
                handled = True
            
            # ESPACIO - BLANCO
            elif key == 32:
                self.current_annotation['tile_letter'] = 'BLANCO'
                self.current_annotation['tile_type'] = self.TILE_TYPES.index('BLANCO')
                print(f"✓ Tipo: BLANCO (idx: {self.current_annotation['tile_type']})")
                self._update_display_now(window_name)
                handled = True
            
            # A-X - Tipo de loseta (minúsculas) - Ahora incluye R, T, M sin conflicto
            elif 97 <= key <= 122:
                letter = chr(key).upper()
                if letter in self.TILE_TYPES and letter != 'Ñ':
                    self.current_annotation['tile_letter'] = letter
                    self.current_annotation['tile_type'] = self.TILE_TYPES.index(letter)
                    print(f"✓ Tipo: {letter} (idx: {self.current_annotation['tile_type']})")
                    self._update_display_now(window_name)
                    handled = True
                else:
                    print(f"✗ '{letter}' no es válido (solo A-X sin Ñ)")
            
            # A-X - Tipo de loseta (mayúsculas) - Ahora incluye R, T, M sin conflicto
            elif 65 <= key <= 90:
                letter = chr(key)
                if letter in self.TILE_TYPES and letter != 'Ñ':
                    self.current_annotation['tile_letter'] = letter
                    self.current_annotation['tile_type'] = self.TILE_TYPES.index(letter)
                    print(f"✓ Tipo: {letter} (idx: {self.current_annotation['tile_type']})")
                    self._update_display_now(window_name)
                    handled = True
                else:
                    print(f"✗ '{letter}' no es válido (solo A-X sin Ñ)")
            
            # Tecla no reconocida
            if not handled and key != 255:
                print(f"[DEBUG] Tecla no reconocida: {key}")
        
        cv2.destroyAllWindows()
        
        # Guardar al finalizar
        if self.annotations:
            self._save_annotations()
            print(f"\n✓ Proceso completado: {len(self.annotations)} losetas anotadas")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Herramienta de anotación de losetas')
    parser.add_argument('tiles_dir', type=str, help='Directorio con imágenes de losetas')
    parser.add_argument('--output', type=str, default='annotations.json',
                       help='Archivo de salida para anotaciones')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.tiles_dir):
        print(f"Error: El directorio '{args.tiles_dir}' no existe")
        return
    
    tool = AnnotationTool(args.tiles_dir, args.output)
    tool.run()


if __name__ == "__main__":
    main()
