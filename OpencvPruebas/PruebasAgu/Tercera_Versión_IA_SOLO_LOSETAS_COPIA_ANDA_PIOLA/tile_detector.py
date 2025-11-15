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
import torch
import torchvision.transforms as transforms
from carcassonne_cnn import CarcassonneCNN


class CarcassonneTileDetector:
    """Detector de losetas usando referencias visuales oficiales"""

    def __init__(self, reference_folder: str = "referencias_organizadas"):
        self.reference_folder = Path(reference_folder)
        self.references = {}  # Ahora será un dict de listas de imágenes
        # Losetas que tienen escudo según reglas del juego
        self.tiles_with_shields = {'C', 'F', 'M', 'O', 'Q', 'S'}
        # TODAS las losetas pueden tener meeple
        self.tiles_with_possible_meeple = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # Todas las letras
        # Imagen de referencia del escudo
        self.shield_template = None
        # Modelo CNN
        self.cnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Mapear índices a letras (0-23 = A-X, 24 = BLANCO)
        self.idx_to_letter = {i: chr(65 + i) for i in range(24)}  # 0=A, 1=B, ..., 23=X
        self.idx_to_letter[24] = 'BLANCO'
        self._load_references()
        self._load_cnn_model()

    def _load_references(self):
        """Carga las imágenes de referencia desde carpetas organizadas"""
        if not self.reference_folder.exists():
            raise ValueError(f"Carpeta de referencias no encontrada: {self.reference_folder}")

        print(f"Cargando referencias desde: {self.reference_folder}")

        total_images = 0
        # Cargar una imagen de referencia por letra (la primera PNG encontrada)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            letter_folder = self.reference_folder / letter
            if letter_folder.exists():
                # Buscar la primera imagen PNG en la carpeta
                for img_file in letter_folder.glob("*.png"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.resize(img, (200, 200))  # Tamaño estándar
                        self.references[letter] = img
                        total_images += 1
                        print(f"  [OK] {img_file.name} cargada para {letter}")
                        break  # Solo cargar la primera
                    else:
                        print(f"  [ERROR] Error cargando {img_file.name}")
                else:
                    print(f"  [NOT FOUND] No se encontró imagen PNG para {letter}")
            else:
                print(f"  [NOT FOUND] Carpeta para {letter} no existe")

        # Cargar imagen de referencia del escudo
        shield_path = self.reference_folder / "Shield.png"
        if shield_path.exists():
            self.shield_template = cv2.imread(str(shield_path))
            if self.shield_template is not None:
                print(f"  [OK] Shield.png cargada")
            else:
                print(f"  [ERROR] Error cargando Shield.png")
        else:
            print(f"  [NOT FOUND] Shield.png no encontrada")

        if not self.references:
            raise ValueError("No se pudieron cargar referencias")

        print(f"Total imágenes cargadas: {total_images} en {len(self.references)} clases")

    def _load_cnn_model(self):
        """Carga el modelo CNN entrenado con múltiples imágenes"""
        # Intentar cargar el modelo multi-imagen primero
        model_paths = [
            Path("carcassonne_cnn_multi_model.pth"),
            Path("carcassonne_cnn_model.pth")
        ]

        for model_path in model_paths:
            if model_path.exists():
                try:
                    self.cnn_model = CarcassonneCNN(num_classes=25)
                    self.cnn_model.load_state_dict(torch.load(str(model_path), map_location=self.device))
                    self.cnn_model.to(self.device)
                    self.cnn_model.eval()
                    print(f"  [OK] Modelo CNN cargado desde {model_path.name}")
                    return
                except Exception as e:
                    print(f"  [ERROR] Error cargando {model_path.name}: {e}")
                    continue

        print("  [NOT FOUND] Ningún modelo CNN encontrado, usando método tradicional")
        self.cnn_model = None

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

        # DETECTAR ESCUDO ANTES de eliminar meeple
        has_shield = self._detect_shield_with_template(image)

        # Buscar meeple en TODAS las losetas (cualquier loseta puede tener meeple)
        image = self._remove_meeple(hsv, image)

        # Usar CNN si está disponible
        if self.cnn_model is not None:
            tile_type, confidence = self._detect_with_cnn(image)
            
            # Si confianza baja, combinar con método tradicional
            if confidence < 0.7:
                traditional_type = self._detect_with_traditional(image, has_shield)
                if tile_type == traditional_type:
                    # Si coinciden, usar ese tipo con confianza media
                    confidence = (confidence + 0.7) / 2
                else:
                    # Si no coinciden, usar el de mayor confianza
                    if confidence > 0.5:
                        pass  # Mantener CNN
                    else:
                        tile_type = traditional_type
                        confidence = 0.6  # Confianza del tradicional
            
            # Si la CNN dice BLANCO con alta confianza, verificar con criterios más estrictos
            if tile_type == 'BLANCO' and confidence > 0.8:
                if self._is_blank_image_strict(image):
                    print("Loseta detectada: BLANCO (sin loseta)")
                    return "BLANCO"
                else:
                    # Si no cumple criterios estrictos, usar segunda mejor predicción de CNN
                    tile_type, confidence = self._get_second_best_cnn_prediction(image)
        else:
            tile_type = self._detect_with_traditional(image, has_shield)
            confidence = 0.7  # Confianza fija para método tradicional
        
        # Verificar si es blanco usando criterios simples (solo si CNN no está segura)
        if tile_type != 'BLANCO' and self._is_blank_image(image):
            print("Loseta detectada: BLANCO (sin loseta)")
            return "BLANCO"
        
        # Para confianza baja, devolver la mejor predicción disponible en lugar de DESCONOCIDO
        if confidence < 0.6:
            print(f"Loseta detectada: {tile_type} (confianza baja: {confidence:.2f})")
            return tile_type

        print(f"Loseta detectada: {tile_type} (confianza: {confidence:.2f})")
        return tile_type

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

    def _is_blank_image_strict(self, image: np.ndarray) -> bool:
        """Verifica si la imagen está definitivamente blanca (criterios muy estrictos)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        total_pixels = gray.size
        white_pixels = np.sum(gray > 200)  # Threshold más alto
        white_ratio = white_pixels / total_pixels
        variance = np.var(gray)
        mean_intensity = np.mean(gray)

        # Criterios MUY estrictos para áreas blancas definitivas:
        # - >85% píxeles muy claros (>200)
        # - Intensidad media muy alta (>190)
        # - Varianza muy baja (<500)
        is_blank = (white_ratio > 0.85 and
                   mean_intensity > 190 and
                   variance < 500)

        return is_blank

    def _is_blank_image(self, image: np.ndarray) -> bool:
        """Verifica si la imagen está mayoritariamente blanca (sin loseta)"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calcular estadísticas
        total_pixels = gray.size
        white_pixels = np.sum(gray > 180)  # Threshold más alto
        white_ratio = white_pixels / total_pixels
        variance = np.var(gray)
        mean_intensity = np.mean(gray)

        # Criterios más relajados para áreas blancas:
        # - >60% píxeles muy claros (>180)
        # - Intensidad media alta (>150)
        # - Varianza baja (<2000)
        is_blank = (white_ratio > 0.6 and
                   mean_intensity > 150 and
                   variance < 2000)

        # Debug info
        if is_blank:
            print(f"  Detectado como blanco: ratio {white_ratio:.2f}, media {mean_intensity:.1f}, varianza {variance:.1f}")

        return is_blank

    def _get_second_best_cnn_prediction(self, image: np.ndarray) -> tuple[str, float]:
        """Obtiene la segunda mejor predicción de la CNN cuando la primera es BLANCO"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.cnn_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Obtener top 2 predicciones
            top2_prob, top2_idx = torch.topk(probabilities, 2, dim=1)
            
            # La segunda mejor (excluyendo BLANCO si es la primera)
            second_idx = top2_idx[0][1].item()  # Segundo mejor
            second_prob = top2_prob[0][1].item()
            
            second_tile = self.idx_to_letter.get(second_idx, '?')
            
        return second_tile, second_prob

    def _detect_shield_with_template(self, image: np.ndarray) -> bool:
        """Detecta escudos usando template matching con imagen de referencia"""
        if self.shield_template is None:
            return False

        # Convertir ambas imágenes a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(self.shield_template, cv2.COLOR_BGR2GRAY)

        # Aplicar Gaussian blur para reducir ruido
        gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        gray_template = cv2.GaussianBlur(gray_template, (3, 3), 0)

        # Template matching
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # Threshold ajustado para incluir M pero excluir N
        has_shield = max_val > 0.38  # Bajé a 0.38 para incluir M (0.395)

        # Debug: mostrar resultado
        if has_shield:
            print(f"  Escudo detectado (confianza: {max_val:.2f})")

        return has_shield

    def _detect_with_cnn(self, image: np.ndarray) -> tuple[str, float]:
        """Detecta usando el modelo CNN"""
        # Preprocesar imagen
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            outputs = self.cnn_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_idx = predicted.item()
            confidence_val = confidence.item()

        tile_type = self.idx_to_letter.get(predicted_idx, '?')
        return tile_type, confidence_val

    def _detect_with_traditional(self, image: np.ndarray, has_shield: bool) -> str:
        """Detecta usando método tradicional de comparación de características"""
        # Extraer features de la imagen de entrada
        input_features = self._extract_features(image, has_shield=has_shield)

        # Comparar con cada referencia
        best_match = '?'
        best_score = 0.0

        for letter, ref_img in self.references.items():
            ref_features = self._extract_features(ref_img, letter)  # Pasar la letra para determinar escudo
            score = self._compare_features(input_features, ref_features)
            if score > best_score:
                best_score = score
                best_match = letter

        return best_match

    def _has_possible_meeple(self, image: np.ndarray) -> bool:
        """Verifica si la imagen podría tener meeple (basado en colores púrpura)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Rango para meeple (violeta/púrpura)
        lower_meeple = np.array([120, 30, 30])
        upper_meeple = np.array([160, 255, 255])
        meeple_mask = cv2.inRange(hsv, lower_meeple, upper_meeple)

        # Si hay píxeles púrpura, podría tener meeple
        return np.sum(meeple_mask > 0) > 50  # Threshold mínimo

    def _extract_features(self, image: np.ndarray, tile_letter: str = None, has_shield: bool = None) -> Dict:
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

        # Nueva característica: análisis de contornos para formas
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calcular métricas de contornos
        if contours:
            num_contours = len(contours)
            total_area = sum(cv2.contourArea(c) for c in contours)
            total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
            complexity = total_perimeter / max(total_area, 1)  # Evitar división por cero
        else:
            num_contours = 0
            total_area = 0
            complexity = 0

        # Normalizar métricas de contornos
        contour_features = np.array([
            min(num_contours / 20.0, 1.0),  # Máximo 20 contornos
            min(total_area / 50000.0, 1.0),  # Área máxima esperada
            min(complexity / 0.5, 1.0)  # Complejidad máxima esperada
        ])

        # Determinar si tiene escudo basado en reglas del juego (no detección automática)
        if has_shield is not None:
            final_has_shield = has_shield
        elif tile_letter:
            final_has_shield = tile_letter in self.tiles_with_shields
        else:
            final_has_shield = self._detect_shield_with_template(image)

        return {
            'orb_kp': kp,
            'orb_des': des,
            'histogram': hist,
            'template': template,
            'contour_features': contour_features,
            'has_shield': final_has_shield
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
                score += min(1.0, orb_score) * 30.0  # Aumentado de 25 a 30
            except:
                pass

        # Histograma
        hist_corr = cv2.compareHist(feat1['histogram'], feat2['histogram'], cv2.HISTCMP_CORREL)
        score += max(0, hist_corr) * 30.0  # Aumentado de 25 a 30

        # Template matching - Aumentado significativamente
        result = cv2.matchTemplate(feat1['template'], feat2['template'], cv2.TM_CCOEFF_NORMED)
        score += result[0][0] * 40.0  # Aumentado de 35 a 40

        # Nueva característica: comparación de contornos/formas
        if 'contour_features' in feat1 and 'contour_features' in feat2:
            contour_diff = np.linalg.norm(feat1['contour_features'] - feat2['contour_features'])
            contour_similarity = max(0, 1.0 - contour_diff)  # Convertir distancia a similitud
            score += contour_similarity * 10.0  # Aumentado de 5 a 10

        # ESCUDO - Feature crítico para diferenciar losetas similares
        shield_match = feat1['has_shield'] == feat2['has_shield']
        if shield_match:
            score += 20.0  # Aumentado de 15 a 20
        else:
            score -= 15.0  # Aumentado de 12 a 15

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