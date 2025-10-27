"""
Evaluación completa del modelo CNN de Carcassonne
Genera métricas detalladas, matrices de confusión y análisis de errores
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import os

from carcassonne_cnn import CarcassonneCNN, CarcassonneTileDataset, create_data_transforms


class ModelEvaluator:
    """Evaluador completo del modelo"""
    
    def __init__(self, model_path: str, test_annotations: str, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Cargar modelo
        self.model = CarcassonneCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Cargar dataset de test
        test_dataset = CarcassonneTileDataset(
            test_annotations,
            transform=create_data_transforms(augment=False)
        )
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Resultados
        self.predictions = {
            'tile_type': [],
            'rotation': [],
            'has_meeple': [],
            'meeple_position': []
        }
        self.ground_truth = {
            'tile_type': [],
            'rotation': [],
            'has_meeple': [],
            'meeple_position': []
        }
        self.confidences = {
            'tile_type': [],
            'rotation': []
        }
    
    def evaluate(self):
        """Realiza la evaluación completa"""
        print("="*60)
        print("EVALUANDO MODELO")
        print("="*60)
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluando"):
                images = images.to(self.device)
                outputs = self.model(images)
                
                # Tile type
                tile_probs = torch.softmax(outputs['tile_type'], dim=1)
                tile_pred = torch.argmax(tile_probs, dim=1)
                tile_conf = torch.max(tile_probs, dim=1)[0]
                
                self.predictions['tile_type'].extend(tile_pred.cpu().numpy())
                self.ground_truth['tile_type'].extend(labels['tile_type'].numpy())
                self.confidences['tile_type'].extend(tile_conf.cpu().numpy())
                
                # Rotation
                rot_probs = torch.softmax(outputs['rotation'], dim=1)
                rot_pred = torch.argmax(rot_probs, dim=1)
                rot_conf = torch.max(rot_probs, dim=1)[0]
                
                self.predictions['rotation'].extend(rot_pred.cpu().numpy())
                self.ground_truth['rotation'].extend(labels['rotation'].numpy())
                self.confidences['rotation'].extend(rot_conf.cpu().numpy())
                
                # Has meeple
                meeple_pred = (outputs['has_meeple'].squeeze() > 0.5).float()
                self.predictions['has_meeple'].extend(meeple_pred.cpu().numpy())
                self.ground_truth['has_meeple'].extend(labels['has_meeple'].numpy())
                
                # Meeple position
                pos_pred = torch.argmax(outputs['meeple_position'], dim=1) - 1
                self.predictions['meeple_position'].extend(pos_pred.cpu().numpy())
                self.ground_truth['meeple_position'].extend(
                    (labels['meeple_position'] - 1).numpy()
                )
        
        print("✓ Evaluación completada\n")
    
    def compute_metrics(self) -> Dict:
        """Calcula todas las métricas"""
        metrics = {}
        
        # Tile type metrics
        tile_acc = accuracy_score(
            self.ground_truth['tile_type'],
            self.predictions['tile_type']
        )
        tile_prec, tile_rec, tile_f1, _ = precision_recall_fscore_support(
            self.ground_truth['tile_type'],
            self.predictions['tile_type'],
            average='weighted'
        )
        
        metrics['tile_type'] = {
            'accuracy': tile_acc,
            'precision': tile_prec,
            'recall': tile_rec,
            'f1_score': tile_f1,
            'avg_confidence': np.mean(self.confidences['tile_type'])
        }
        
        # Rotation metrics
        rot_acc = accuracy_score(
            self.ground_truth['rotation'],
            self.predictions['rotation']
        )
        rot_prec, rot_rec, rot_f1, _ = precision_recall_fscore_support(
            self.ground_truth['rotation'],
            self.predictions['rotation'],
            average='weighted'
        )
        
        metrics['rotation'] = {
            'accuracy': rot_acc,
            'precision': rot_prec,
            'recall': rot_rec,
            'f1_score': rot_f1,
            'avg_confidence': np.mean(self.confidences['rotation'])
        }
        
        # Meeple detection metrics
        meeple_acc = accuracy_score(
            self.ground_truth['has_meeple'],
            self.predictions['has_meeple']
        )
        meeple_prec, meeple_rec, meeple_f1, _ = precision_recall_fscore_support(
            self.ground_truth['has_meeple'],
            self.predictions['has_meeple'],
            average='binary'
        )
        
        metrics['has_meeple'] = {
            'accuracy': meeple_acc,
            'precision': meeple_prec,
            'recall': meeple_rec,
            'f1_score': meeple_f1
        }
        
        # Meeple position metrics (solo para casos con ficha)
        has_meeple_mask = np.array(self.ground_truth['has_meeple']) == 1
        if has_meeple_mask.sum() > 0:
            pos_acc = accuracy_score(
                np.array(self.ground_truth['meeple_position'])[has_meeple_mask],
                np.array(self.predictions['meeple_position'])[has_meeple_mask]
            )
            metrics['meeple_position'] = {
                'accuracy': pos_acc,
                'n_samples': has_meeple_mask.sum()
            }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Imprime métricas en formato legible"""
        print("="*60)
        print("MÉTRICAS DE EVALUACIÓN")
        print("="*60)
        
        print("\n🎯 TIPO DE LOSETA:")
        print(f"  Accuracy:   {metrics['tile_type']['accuracy']:.4f} ({metrics['tile_type']['accuracy']*100:.2f}%)")
        print(f"  Precision:  {metrics['tile_type']['precision']:.4f}")
        print(f"  Recall:     {metrics['tile_type']['recall']:.4f}")
        print(f"  F1-Score:   {metrics['tile_type']['f1_score']:.4f}")
        print(f"  Confianza:  {metrics['tile_type']['avg_confidence']:.4f}")
        
        print("\n🔄 ROTACIÓN:")
        print(f"  Accuracy:   {metrics['rotation']['accuracy']:.4f} ({metrics['rotation']['accuracy']*100:.2f}%)")
        print(f"  Precision:  {metrics['rotation']['precision']:.4f}")
        print(f"  Recall:     {metrics['rotation']['recall']:.4f}")
        print(f"  F1-Score:   {metrics['rotation']['f1_score']:.4f}")
        print(f"  Confianza:  {metrics['rotation']['avg_confidence']:.4f}")
        
        print("\n👤 DETECCIÓN DE FICHA:")
        print(f"  Accuracy:   {metrics['has_meeple']['accuracy']:.4f} ({metrics['has_meeple']['accuracy']*100:.2f}%)")
        print(f"  Precision:  {metrics['has_meeple']['precision']:.4f}")
        print(f"  Recall:     {metrics['has_meeple']['recall']:.4f}")
        print(f"  F1-Score:   {metrics['has_meeple']['f1_score']:.4f}")
        
        if 'meeple_position' in metrics:
            print("\n📍 POSICIÓN DE FICHA:")
            print(f"  Accuracy:   {metrics['meeple_position']['accuracy']:.4f} ({metrics['meeple_position']['accuracy']*100:.2f}%)")
            print(f"  Muestras:   {metrics['meeple_position']['n_samples']}")
    
    def plot_confusion_matrices(self, output_dir: str = '.'):
        """Genera matrices de confusión"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Confusion matrix - Tile Type
        cm_tile = confusion_matrix(
            self.ground_truth['tile_type'],
            self.predictions['tile_type']
        )
        
        sns.heatmap(cm_tile, annot=True, fmt='d', cmap='Blues', 
                   ax=axes[0], cbar_kws={'label': 'Cantidad'})
        axes[0].set_title('Matriz de Confusión - Tipo de Loseta', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicción')
        axes[0].set_ylabel('Real')
        
        # Confusion matrix - Rotation
        cm_rot = confusion_matrix(
            self.ground_truth['rotation'],
            self.predictions['rotation']
        )
        
        rotation_labels = ['0°', '90°', '180°', '270°']
        sns.heatmap(cm_rot, annot=True, fmt='d', cmap='Greens',
                   xticklabels=rotation_labels, yticklabels=rotation_labels,
                   ax=axes[1], cbar_kws={'label': 'Cantidad'})
        axes[1].set_title('Matriz de Confusión - Rotación',
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicción')
        axes[1].set_ylabel('Real')
        
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        print(f"✓ Matrices de confusión guardadas en {output_path / 'confusion_matrices.png'}")
        plt.show()
    
    def analyze_errors(self, output_file: str = 'error_analysis.json'):
        """Analiza los errores del modelo"""
        errors = {
            'tile_type_errors': [],
            'rotation_errors': [],
            'meeple_errors': []
        }
        
        # Errores de tipo
        for i, (pred, true) in enumerate(zip(
            self.predictions['tile_type'],
            self.ground_truth['tile_type']
        )):
            if pred != true:
                errors['tile_type_errors'].append({
                    'index': i,
                    'predicted': int(pred),
                    'true': int(true),
                    'confidence': float(self.confidences['tile_type'][i])
                })
        
        # Errores de rotación
        for i, (pred, true) in enumerate(zip(
            self.predictions['rotation'],
            self.ground_truth['rotation']
        )):
            if pred != true:
                errors['rotation_errors'].append({
                    'index': i,
                    'predicted': int(pred) * 90,
                    'true': int(true) * 90,
                    'confidence': float(self.confidences['rotation'][i])
                })
        
        # Errores de ficha
        for i, (pred, true) in enumerate(zip(
            self.predictions['has_meeple'],
            self.ground_truth['has_meeple']
        )):
            if pred != true:
                errors['meeple_errors'].append({
                    'index': i,
                    'predicted': bool(pred),
                    'true': bool(true)
                })
        
        # Guardar análisis
        with open(output_file, 'w') as f:
            json.dump(errors, f, indent=2)
        
        # Imprimir resumen
        print("\n" + "="*60)
        print("ANÁLISIS DE ERRORES")
        print("="*60)
        print(f"\nErrores de tipo de loseta: {len(errors['tile_type_errors'])}")
        print(f"Errores de rotación: {len(errors['rotation_errors'])}")
        print(f"Errores de detección de ficha: {len(errors['meeple_errors'])}")
        
        # Analizar patrones de error
        if errors['tile_type_errors']:
            print("\nTipos más confundidos:")
            confusion_pairs = {}
            for err in errors['tile_type_errors']:
                pair = (err['true'], err['predicted'])
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            
            top_confusions = sorted(confusion_pairs.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            for (true, pred), count in top_confusions:
                print(f"  Tipo {true} → {pred}: {count} veces")
        
        print(f"\n✓ Análisis de errores guardado en {output_file}")
    
    def plot_confidence_distribution(self, output_dir: str = '.'):
        """Grafica distribución de confianzas"""
        output_path = Path(output_dir)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Tile type confidence
        correct_tile = np.array(self.predictions['tile_type']) == np.array(self.ground_truth['tile_type'])
        conf_correct = np.array(self.confidences['tile_type'])[correct_tile]
        conf_incorrect = np.array(self.confidences['tile_type'])[~correct_tile]
        
        axes[0].hist(conf_correct, bins=20, alpha=0.7, label='Correctas', color='green')
        axes[0].hist(conf_incorrect, bins=20, alpha=0.7, label='Incorrectas', color='red')
        axes[0].set_xlabel('Confianza')
        axes[0].set_ylabel('Frecuencia')
        axes[0].set_title('Distribución de Confianza - Tipo de Loseta', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Rotation confidence
        correct_rot = np.array(self.predictions['rotation']) == np.array(self.ground_truth['rotation'])
        conf_rot_correct = np.array(self.confidences['rotation'])[correct_rot]
        conf_rot_incorrect = np.array(self.confidences['rotation'])[~correct_rot]
        
        axes[1].hist(conf_rot_correct, bins=20, alpha=0.7, label='Correctas', color='blue')
        axes[1].hist(conf_rot_incorrect, bins=20, alpha=0.7, label='Incorrectas', color='orange')
        axes[1].set_xlabel('Confianza')
        axes[1].set_ylabel('Frecuencia')
        axes[1].set_title('Distribución de Confianza - Rotación', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_distribution.png', dpi=150, bbox_inches='tight')
        print(f"✓ Distribución de confianza guardada en {output_path / 'confidence_distribution.png'}")
        plt.show()
    
    def generate_report(self, output_dir: str = 'evaluation_results'):
        """Genera reporte completo de evaluación"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("GENERANDO REPORTE DE EVALUACIÓN")
        print("="*60)
        
        # Calcular métricas
        metrics = self.compute_metrics()
        
        # Imprimir métricas
        self.print_metrics(metrics)

        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            else:
                return obj

        metrics = make_serializable(metrics)

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Guardar métricas en JSON
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Métricas guardadas en {metrics_file}")
        
        # Generar visualizaciones
        self.plot_confusion_matrices(output_path)
        self.plot_confidence_distribution(output_path)
        
        # Análisis de errores
        self.analyze_errors(output_path / 'error_analysis.json')
        
        print("\n" + "="*60)
        print("✓ REPORTE COMPLETO GENERADO")
        print("="*60)
        print(f"Directorio: {output_path}")


def main():
    """Función principal"""
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python evaluate_model.py <model.pth> <test_annotations.json> [output_dir]")
        print("\nEjemplo:")
        print("  python evaluate_model.py best_model.pth test_annotations.json results/")
        return
    
    model_path = sys.argv[1]
    test_annotations = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'evaluation_results'
    
    # Crear evaluador
    evaluator = ModelEvaluator(model_path, test_annotations)
    
    # Evaluar
    evaluator.evaluate()
    
    # Generar reporte
    evaluator.generate_report(output_dir)


if __name__ == "__main__":
    main()
