#!/usr/bin/env python3
"""
Evaluador del detector usando anotaciones manuales ground truth
Compara detección automática vs anotaciones manuales y sugiere mejoras
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from src.meeple_detector_cv import MeepleDetector
import matplotlib.pyplot as plt

class DetectorEvaluator:
    """Evalúa el rendimiento del detector usando ground truth"""

    def __init__(self, ground_truth_file: str):
        self.ground_truth = self.load_ground_truth(ground_truth_file)
        self.detector = MeepleDetector()
        self.results = {}

    def load_ground_truth(self, file_path: str) -> Dict:
        """Cargar anotaciones ground truth"""
        if not Path(file_path).exists():
            print(f"❌ Archivo ground truth no encontrado: {file_path}")
            return {}

        with open(file_path, 'r') as f:
            return json.load(f)

    def evaluate_image(self, image_path: str) -> Dict:
        """Evaluar una imagen específica"""
        if image_path not in self.ground_truth:
            return {'error': f'No hay ground truth para {image_path}'}

        # Obtener detección automática
        auto_result = self.detector.process_image(image_path)
        if 'error' in auto_result:
            return auto_result

        # Obtener ground truth
        gt_meeples = self.ground_truth[image_path]

        # Comparar
        auto_meeples = auto_result['meeples']

        # Métricas básicas
        gt_count = len(gt_meeples)
        auto_count = len(auto_meeples)

        # Calcular precisión de color y posición
        correct_color = 0
        correct_position = 0
        false_positives = 0
        false_negatives = 0

        # Para cada meeple detectado automáticamente
        for auto_meeple in auto_meeples:
            found_match = False
            for gt_meeple in gt_meeples:
                if (auto_meeple['color'] == gt_meeple['color'] and
                    auto_meeple['position'] == gt_meeple['position']):
                    correct_color += 1
                    correct_position += 1
                    found_match = True
                    break
            if not found_match:
                false_positives += 1

        # Meeples ground truth no detectados
        false_negatives = gt_count - correct_position

        return {
            'image_path': image_path,
            'ground_truth_count': gt_count,
            'auto_count': auto_count,
            'correct_color': correct_color,
            'correct_position': correct_position,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': correct_position / auto_count if auto_count > 0 else 0,
            'recall': correct_position / gt_count if gt_count > 0 else 0,
            'auto_meeples': auto_meeples,
            'gt_meeples': gt_meeples
        }

    def evaluate_all(self) -> Dict:
        """Evaluar todas las imágenes con ground truth"""
        results = {}
        total_images = len(self.ground_truth)

        print(f"🔍 Evaluando {total_images} imágenes con ground truth...")
        print("=" * 60)

        total_gt = 0
        total_auto = 0
        total_correct_pos = 0
        total_correct_color = 0
        total_fp = 0
        total_fn = 0

        for image_path in self.ground_truth.keys():
            result = self.evaluate_image(image_path)
            results[image_path] = result

            if 'error' not in result:
                print(f"📊 {Path(image_path).name}:")
                print(f"   GT: {result['ground_truth_count']} | Auto: {result['auto_count']}")
                print(f"   Correctos (posición): {result['correct_position']}")
                print(f"   Precisión: {result['precision']:.2f} | Recall: {result['recall']:.2f}")
                print()

                total_gt += result['ground_truth_count']
                total_auto += result['auto_count']
                total_correct_pos += result['correct_position']
                total_correct_color += result['correct_color']
                total_fp += result['false_positives']
                total_fn += result['false_negatives']

        # Métricas globales
        overall_precision = total_correct_pos / total_auto if total_auto > 0 else 0
        overall_recall = total_correct_pos / total_gt if total_gt > 0 else 0
        f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

        summary = {
            'total_images': total_images,
            'total_ground_truth_meeples': total_gt,
            'total_auto_meeples': total_auto,
            'total_correct_position': total_correct_pos,
            'total_correct_color': total_correct_color,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'f1_score': f1_score,
            'results': results
        }

        print("📈 RESULTADOS GLOBALES:")
        print("=" * 40)
        print(f"Imágenes evaluadas: {total_images}")
        print(f"Meeples ground truth: {total_gt}")
        print(f"Meeples detectados: {total_auto}")
        print(f"Correctos (posición): {total_correct_pos}")
        print(f"False positives: {total_fp}")
        print(f"False negatives: {total_fn}")
        print(f"Precisión: {overall_precision:.3f}")
        print(f"Recall: {overall_recall:.3f}")
        print(f"F1-Score: {f1_score:.3f}")

        return summary

    def visualize_comparison(self, image_path: str, save_path: Optional[str] = None):
        """Visualizar comparación entre detección automática y ground truth"""
        if image_path not in self.ground_truth:
            print(f"❌ No hay ground truth para {image_path}")
            return

        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ No se pudo cargar {image_path}")
            return

        # Obtener resultados
        eval_result = self.evaluate_image(image_path)

        # Crear visualización lado a lado
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Ground Truth
        gt_display = image.copy()
        h, w = gt_display.shape[:2]

        # Dibujar grid
        for i in range(1, 3):
            cv2.line(gt_display, (int(w*i/3), 0), (int(w*i/3), h), (255, 255, 255), 1)
            cv2.line(gt_display, (0, int(h*i/3)), (w, int(h*i/3)), (255, 255, 255), 1)

        # Dibujar GT meeples
        for gt_meeple in eval_result['gt_meeples']:
            pos = gt_meeple['position']
            grid_y, grid_x = divmod(pos, 3)
            center_x = int((grid_x + 0.5) * w / 3)
            center_y = int((grid_y + 0.5) * h / 3)

            color = (255, 0, 0) if gt_meeple['color'] == 'blue' else (0, 0, 0)
            cv2.circle(gt_display, (center_x, center_y), 20, color, 3)
            cv2.putText(gt_display, f"GT-{gt_meeple['color'][0].upper()}{pos}",
                       (center_x-15, center_y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ax1.imshow(cv2.cvtColor(gt_display, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Ground Truth\n{len(eval_result["gt_meeples"])} meeples')
        ax1.axis('off')

        # Detección automática
        auto_display = image.copy()

        # Dibujar grid
        for i in range(1, 3):
            cv2.line(auto_display, (int(w*i/3), 0), (int(w*i/3), h), (255, 255, 255), 1)
            cv2.line(auto_display, (0, int(h*i/3)), (w, int(h*i/3)), (255, 255, 255), 1)

        # Dibujar detección automática
        for auto_meeple in eval_result['auto_meeples']:
            x, y, r = auto_meeple['circle']
            color = (255, 0, 0) if auto_meeple['color'] == 'blue' else (0, 0, 0)
            cv2.circle(auto_display, (x, y), r, color, 2)
            cv2.putText(auto_display, f"Auto-{auto_meeple['color'][0].upper()}{auto_meeple['position']}",
                       (x-15, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ax2.imshow(cv2.cvtColor(auto_display, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Detección Automática\n{len(eval_result["auto_meeples"])} meeples\nPrec: {eval_result["precision"]:.2f}, Rec: {eval_result["recall"]:.2f}')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"💾 Comparación guardada: {save_path}")
        else:
            plt.show()

def main():
    """Función principal"""
    evaluator = DetectorEvaluator('manual_annotations.json')

    if not evaluator.ground_truth:
        print("❌ No hay anotaciones ground truth. Ejecuta primero: python meeple_annotator.py")
        return

    # Evaluar todas las imágenes
    summary = evaluator.evaluate_all()

    # Guardar resultados
    with open('evaluation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("💾 Resultados guardados en: evaluation_results.json")

    # Preguntar si quiere visualizar algunas comparaciones
    print("\n🔍 ¿Quieres visualizar algunas comparaciones específicas?")
    response = input("Ingresa nombre de imagen (o 'no' para salir): ").strip()

    while response.lower() != 'no':
        image_path = f'real_test_images/{response}'
        if Path(image_path).exists():
            evaluator.visualize_comparison(image_path)
        else:
            print(f"❌ Imagen no encontrada: {image_path}")

        response = input("Otra imagen (o 'no' para salir): ").strip()

if __name__ == "__main__":
    main()