#!/usr/bin/env python3
"""
Prueba rápida del detector actualizado con rangos HSV precisos
"""

from src.meeple_detector_cv import MeepleDetector
import json

def main():
    # Probar con algunas imágenes específicas
    detector = MeepleDetector()

    test_images = [
        'real_test_images/A20251113_185222.jpg',  # Primera A
        'real_test_images/A20251113_185519.jpg',  # Última A
        'real_test_images/B20251113_185604.jpg',  # Primera B
        'real_test_images/B20251113_185956.jpg',  # Última B
    ]

    print('Probando detector actualizado con rangos HSV precisos:')
    print('=' * 60)

    for img_path in test_images:
        try:
            result = detector.process_image(img_path)
            if 'error' not in result:
                filename = img_path.split('/')[-1]
                print(f'{filename}:')
                print(f'  Meeples encontrados: {result["meeples_found"]}')
                for meeple in result['meeples']:
                    print(f'    Color: {meeple["color"]}, Posición: {meeple["position"]}')
            else:
                filename = img_path.split('/')[-1]
                print(f'{filename}: ERROR - {result["error"]}')
        except Exception as e:
            filename = img_path.split('/')[-1]
            print(f'{filename}: ERROR - {str(e)}')
        print()

if __name__ == "__main__":
    main()