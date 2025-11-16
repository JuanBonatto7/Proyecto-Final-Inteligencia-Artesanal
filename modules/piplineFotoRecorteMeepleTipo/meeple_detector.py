#!/usr/bin/env python3
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path


class MeepleDetector:
    def __init__(self):
        pass

    def detect_meeple_by_color(self, image: np.ndarray) -> Dict:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        best_detection = None
        best_score = 0

        # Azul
        lower_blue = np.array([70, 25, 30])
        upper_blue = np.array([150, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        blue_result = self._analyze_mask(blue_mask, image, 'blue', w, h)
        if blue_result and blue_result['score'] > best_score:
            best_score = blue_result['score']
            best_detection = blue_result

        # Negro
        black_result = self._detect_black_meeple_improved(image, hsv, w, h)
        if black_result and black_result['score'] > best_score:
            best_score = black_result['score']
            best_detection = black_result

        return best_detection

    def _detect_black_meeple_improved(self, image: np.ndarray, hsv: np.ndarray, w: int, h: int) -> Optional[Dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, black_mask_gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        lower_black_hsv = np.array([0, 0, 0])
        upper_black_hsv = np.array([180, 100, 60])
        black_mask_hsv = cv2.inRange(hsv, lower_black_hsv, upper_black_hsv)

        black_mask = cv2.bitwise_or(black_mask_gray, black_mask_hsv)

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)

        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)

        return self._analyze_mask_strict(black_mask, image, 'black', w, h)

    def _analyze_mask_strict(self, mask: np.ndarray, image: np.ndarray, color_name: str, w: int, h: int) -> Optional[Dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_contour = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = (w * h) * 0.04
            max_area = (w * h) * 0.55
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.4:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bh == 0:
                continue
            aspect_ratio = float(bw) / bh
            if aspect_ratio < 0.5 or aspect_ratio > 1.6:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            bbox_center_x = x + bw // 2
            bbox_center_y = y + bh // 2
            center_offset = np.sqrt((cx - bbox_center_x)**2 + (cy - bbox_center_y)**2)
            max_offset = min(bw, bh) * 0.3
            if center_offset > max_offset:
                continue

            mask_contour = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour], -1, 255, -1)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pixels_in_contour = gray[mask_contour > 0]
            if len(pixels_in_contour) == 0:
                continue

            mean_intensity = np.mean(pixels_in_contour)
            if color_name == 'black' and mean_intensity > 70:
                continue

            std_intensity = np.std(pixels_in_contour)
            if std_intensity > 30:
                continue

            score = (area * 0.4) * (circularity * 0.6)

            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            return None

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

    def _analyze_mask(self, mask: np.ndarray, image: np.ndarray, color_name: str, w: int, h: int) -> Optional[Dict]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_contour = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = (w * h) * 0.015
            max_area = (w * h) * 0.65

            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.2:
                continue

            score = area * 1.5 * circularity

            if score > best_score:
                best_score = score
                best_contour = contour

        if best_contour is None:
            return None

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
        x, y = center
        h, w = image_shape

        cell_w = w / 3.0
        cell_h = h / 3.0

        col = 0 if x < cell_w else (1 if x < cell_w * 2 else 2)
        row = 0 if y < cell_h else (1 if y < cell_h * 2 else 2)

        return row * 3 + col

    def detect_meeple(self, image_path: str) -> Dict:
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

        position = self.get_grid_position(detection['center'], (h, w))
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
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: No se pudo cargar {image_path}")
            return

        h, w = image.shape[:2]
        result = self.detect_meeple(image_path)

        cell_w = w // 3
        cell_h = h // 3

        for i in range(1, 3):
            cv2.line(image, (cell_w * i, 0), (cell_w * i, h), (255, 255, 255), 2)
            cv2.line(image, (0, cell_h * i), (w, cell_h * i), (255, 255, 255), 2)

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
        else:
            cv2.imshow('Detección de Meeple', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    import sys
    import os

    if len(sys.argv) < 2:
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Archivo no encontrado: {image_path}")
        return

    detector = MeepleDetector()
    result = detector.detect_meeple(image_path)

    if 'error' in result:
        print(result['error'])
        return

    if result['has_meeple']:

        if result['circle']:
            x, y, r = result['circle']

    output_path = f"deteccion_{Path(image_path).stem}.jpg"
    detector.visualize_detection(image_path, output_path)


if __name__ == "__main__":
    main()
