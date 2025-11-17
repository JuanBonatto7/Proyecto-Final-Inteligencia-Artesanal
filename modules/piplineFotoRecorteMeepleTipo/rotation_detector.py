"""
Detector de rotación de losetas de Carcassonne.
Detecta la rotación (0°, 90°, 180°, 270°) de una loseta dado su tipo.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict
import sys
import os


class CarcassonneRotationDetector:
    """Detector de rotación usando referencias visuales oficiales"""

    def __init__(self, reference_folder: str = "referencias_organizadas"):
        self.reference_folder = Path(__file__).parent / reference_folder
        self.references = {}  # Dict de imágenes de referencia (0°)
        self._load_references()

    def _load_references(self):
        """Carga las imágenes de referencia desde carpetas organizadas"""
        if not self.reference_folder.exists():
            raise ValueError(f"Carpeta de referencias no encontrada: {self.reference_folder}")

        total_images = 0
        # Cargar una imagen de referencia por letra (la primera PNG encontrada)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWX":
            letter_folder = self.reference_folder / letter
            if letter_folder.exists():
                # Buscar la primera imagen PNG en la carpeta
                for img_file in letter_folder.glob("*.png"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.resize(img, (200, 200))  # Tamaño estándar
                        self.references[letter] = img
                        total_images += 1
                        break  # Solo cargar la primera
                    else:
                        print(f"  [ERROR] Error cargando {img_file.name}")
                else:
                    print(f"  [NOT FOUND] No se encontró imagen PNG para {letter}")
            else:
                print(f"  [NOT FOUND] Carpeta para {letter} no existe")

        if not self.references:
            raise ValueError("No se pudieron cargar referencias")

    def detect_rotation(self, tile_type: str, image_path: str) -> int:
        """Detecta la rotación de la loseta"""

        if tile_type == 'BLANCO':
            return 0

        if tile_type not in self.references:
            print(f"  [WARNING] No hay referencia para {tile_type}, rotación 0°")
            return 0

        # Cargar imagen de entrada
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # Preprocesar igual que en tile_detector
        image = cv2.resize(image, (200, 200))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Remover meeple si existe (igual que en tile_detector)
        lower_meeple = np.array([120, 30, 30])
        upper_meeple = np.array([160, 255, 255])
        meeple_mask = cv2.inRange(hsv, lower_meeple, upper_meeple)

        if np.sum(meeple_mask > 0) > 0:
            kernel = np.ones((5, 5), np.uint8)
            meeple_mask = cv2.dilate(meeple_mask, kernel, iterations=2)
            image = cv2.inpaint(image, meeple_mask, 3, cv2.INPAINT_TELEA)

        # Detectar rotación
        ref_img = self.references[tile_type]
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        best_angle = 0
        best_score = -1

        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated_ref = gray_ref
            elif angle == 90:
                rotated_ref = cv2.rotate(gray_ref, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                rotated_ref = cv2.rotate(gray_ref, cv2.ROTATE_180)
            elif angle == 270:
                rotated_ref = cv2.rotate(gray_ref, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Template matching
            result = cv2.matchTemplate(gray_image, rotated_ref, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_angle = angle
        return best_angle


def main():
    if len(sys.argv) < 3:
        sys.exit(1)

    tile_type = sys.argv[1].upper()
    image_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Error: No existe el archivo {image_path}")
        sys.exit(1)

    try:
        detector = CarcassonneRotationDetector()
        rotation = detector.detect_rotation(tile_type, image_path)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
