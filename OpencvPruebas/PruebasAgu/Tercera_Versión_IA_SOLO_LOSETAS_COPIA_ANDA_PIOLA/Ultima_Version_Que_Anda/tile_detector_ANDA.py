"""
IA Autónoma para detectar losetas de Carcassonne.
Usa matching de imágenes con referencias oficiales para identificar tipos de losetas.
No requiere entrenamiento humano - compara con ejemplos visuales.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import sys
import os


class CarcassonneTileDetector:
    """Detector de losetas usando referencias visuales oficiales"""

    def __init__(self, reference_folder: str = "referencias"):
        self.reference_folder = Path(reference_folder)
        self.references = {}
        self._load_references()

    def _load_references(self):
        """Carga las imágenes de referencia oficiales"""
        if not self.reference_folder.exists():
            raise ValueError(f"Carpeta de referencias no encontrada: {self.reference_folder}")

        print(f"Cargando referencias desde: {self.reference_folder}")

        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            ref_path = self.reference_folder / f"{letter}.png"
            if ref_path.exists():
                img = cv2.imread(str(ref_path))
                if img is not None:
                    img = cv2.resize(img, (200, 200))  # Tamaño estándar
                    self.references[letter] = img
                    print(f"  ✓ {letter}.png cargada")
                else:
                    print(f"  ✗ Error cargando {letter}.png")
            else:
                print(f"  - {letter}.png no encontrada")

        if not self.references:
            raise ValueError("No se pudieron cargar referencias")

        print(f"Total referencias cargadas: {len(self.references)}")

    def detect_tile(self, image_path: str) -> str:
        """Detecta qué tipo de loseta es la imagen"""
        print(f"Analizando loseta: {image_path}")

        # Cargar imagen de entrada
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # Preprocesar
        image = cv2.resize(image, (200, 200))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Remover meeple si existen
        image = self._remove_meeple(hsv, image)

        # Extraer features de la imagen de entrada
        input_features = self._extract_features(image)

        # Comparar con cada referencia
        best_match = '?'
        best_score = 0.0

        for letter, ref_img in self.references.items():
            ref_features = self._extract_features(ref_img)
            score = self._compare_features(input_features, ref_features)
            if score > best_score:
                best_score = score
                best_match = letter

        confidence = min(1.0, best_score / 100.0)  # Normalizar a 0-1
        print(f"Loseta detectada: {best_match} (confianza: {confidence:.2f})")
        return best_match

    def _remove_meeple(self, hsv: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Remueve meeple de la imagen usando inpainting"""
        # Rango para meeple (violeta/púrpura)
        lower_meeple = np.array([120, 30, 30])
        upper_meeple = np.array([160, 255, 255])
        meeple_mask = cv2.inRange(hsv, lower_meeple, upper_meeple)

        if np.sum(meeple_mask > 0) > 0:
            # Dilatar para cubrir bordes
            kernel = np.ones((5, 5), np.uint8)
            meeple_mask = cv2.dilate(meeple_mask, kernel, iterations=2)
            # Inpainting
            image = cv2.inpaint(image, meeple_mask, 3, cv2.INPAINT_TELEA)
            print("  Meeple detectado y removido")

        return image

    def _extract_features(self, image: np.ndarray) -> Dict:
        """Extrae características ORB de la imagen"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ORB detector
        orb = cv2.ORB_create(nfeatures=500, fastThreshold=5)
        kp, des = orb.detectAndCompute(gray, None)

        # Histograma de color HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist = np.concatenate([
            cv2.normalize(hist_h, hist_h).flatten(),
            cv2.normalize(hist_s, hist_s).flatten()
        ])

        # Template reducido
        template = cv2.resize(gray, (64, 64))

        return {
            'orb_kp': kp,
            'orb_des': des,
            'histogram': hist,
            'template': template
        }

    def _compare_features(self, feat1: Dict, feat2: Dict) -> float:
        """Compara features entre dos imágenes"""
        score = 0.0

        # ORB matching
        if feat1['orb_des'] is not None and feat2['orb_des'] is not None:
            try:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(feat1['orb_des'], feat2['orb_des'], k=2)

                good_matches = []
                for m, n in matches:
                    if len([m, n]) == 2 and m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                orb_score = len(good_matches) / 50.0  # Normalizar
                score += min(1.0, orb_score) * 50.0
            except:
                pass

        # Histograma
        hist_corr = cv2.compareHist(feat1['histogram'], feat2['histogram'], cv2.HISTCMP_CORREL)
        score += max(0, hist_corr) * 30.0

        # Template matching
        result = cv2.matchTemplate(feat1['template'], feat2['template'], cv2.TM_CCOEFF_NORMED)
        score += result[0][0] * 20.0

        return score


def main():
    if len(sys.argv) < 2:
        print("Uso: python tile_detector.py <ruta_imagen_loseta>")
        print("Ejemplo: python tile_detector.py loseta.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: No existe el archivo {image_path}")
        sys.exit(1)

    try:
        detector = CarcassonneTileDetector()
        tile_type = detector.detect_tile(image_path)
        print(f"\nResultado: Loseta tipo {tile_type}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()