"""
Utilidades para Evaluación, Métricas y Análisis de Errores
"""

import os
import json
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import CarcassonneCNN
from dataset import CarcassonneDataset, create_dataloaders


class ModelEvaluator:
    """Clase para evaluar el modelo y generar métricas."""
    
    def __init__(
        self,
        model: CarcassonneCNN,
        device: torch.device
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict:
        """
        Evalúa el modelo en un dataset.
        
        Returns:
            Diccionario con métricas detalladas
        """
        # Almacenar predicciones y ground truth
        all_predictions = {
            'tile_type': [],
            'rotation': [],
            'meeple_presence': [],
            'meeple_position': []
        }
        
        all_targets = {
            'tile_type': [],
            'rotation': [],
            'meeple_presence': [],
            'meeple_position': []
        }
        
        # Iterar sobre el dataset
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Predicción
            outputs = self.model(images)
            
            # Guardar predicciones
            for key in all_predictions.keys():
                preds = torch.argmax(outputs[key], dim=1).cpu().numpy()
                targets = labels[key].cpu().numpy()
                
                all_predictions[key].extend(preds)
                all_targets[key].extend(targets)
        
        # Calcular métricas
        metrics = {}
        
        for task in ['tile_type', 'rotation', 'meeple_presence']:
            y_true = np.array(all_targets[task])
            y_pred = np.array(all_predictions[task])
            
            metrics[task] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        
        # Meeple position (solo considerar cuando hay meeple)
        y_true_pos = np.array(all_targets['meeple_position'])
        y_pred_pos = np.array(all_predictions['meeple_position'])
        
        # Filtrar los -1 (sin meeple)
        mask = y_true_pos != -1
        if mask.sum() > 0:
            y_true_pos_filtered = y_true_pos[mask]
            y_pred_pos_filtered = y_pred_pos[mask]
            
            metrics['meeple_position'] = {
                'accuracy': accuracy_score(y_true_pos_filtered, y_pred_pos_filtered),
                'precision': precision_score(y_true_pos_filtered, y_pred_pos_filtered, 
                                            average='weighted', zero_division=0),
                'recall': recall_score(y_true_pos_filtered, y_pred_pos_filtered, 
                                      average='weighted', zero_division=0),
                'f1': f1_score(y_true_pos_filtered, y_pred_pos_filtered, 
                              average='weighted', zero_division=0)
            }
        else:
            metrics['meeple_position'] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # Accuracy general (todas las tareas correctas)
        correct_all = 0
        total = len(all_predictions['tile_type'])
        
        for i in range(total):
            if (all_predictions['tile_type'][i] == all_targets['tile_type'][i] and
                all_predictions['rotation'][i] == all_targets['rotation'][i] and
                all_predictions['meeple_presence'][i] == all_targets['meeple_presence'][i] and
                all_predictions['meeple_position'][i] == all_targets['meeple_position'][i]):
                correct_all += 1
        
        metrics['overall_accuracy'] = correct_all / total
        
        # Guardar predicciones para análisis
        metrics['predictions'] = all_predictions
        metrics['targets'] = all_targets
        
        return metrics
    
    def plot_confusion_matrices(
        self,
        metrics: Dict,
        save_dir: str = 'evaluation_results'
    ):
        """Genera matrices de confusión para cada tarea."""
        os.makedirs(save_dir, exist_ok=True)
        
        tasks = ['tile_type', 'rotation', 'meeple_presence']
        task_names = ['Tipo de Loseta', 'Rotación', 'Presencia de Meeple']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (task, task_name) in enumerate(zip(tasks, task_names)):
            y_true = metrics['targets'][task]
            y_pred = metrics['predictions'][task]
            
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'Matriz de Confusión: {task_name}')
            axes[idx].set_xlabel('Predicción')
            axes[idx].set_ylabel('Verdadero')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matrices de confusión guardadas en {save_path}")
        plt.close()
    
    def generate_error_analysis(
        self,
        metrics: Dict,
        dataset,
        save_path: str = 'evaluation_results/error_analysis.json'
    ):
        """Genera análisis detallado de errores."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        errors = {
            'tile_type': [],
            'rotation': [],
            'meeple_presence': [],
            'meeple_position': []
        }
        
        # Identificar errores
        total = len(metrics['predictions']['tile_type'])
        
        for i in range(total):
            error_entry = {
                'index': i,
                'image_path': dataset.samples[i]['image_path'] if i < len(dataset.samples) else 'unknown'
            }
            
            for task in errors.keys():
                pred = metrics['predictions'][task][i]
                target = metrics['targets'][task][i]
                
                if pred != target:
                    error_entry_copy = error_entry.copy()
                    error_entry_copy['predicted'] = int(pred)
                    error_entry_copy['true'] = int(target)
                    errors[task].append(error_entry_copy)
        
        # Estadísticas de errores
        error_stats = {
            'total_samples': total,
            'errors_by_task': {
                task: {
                    'count': len(error_list),
                    'error_rate': len(error_list) / total,
                    'examples': error_list[:10]  # Top 10 errores
                }
                for task, error_list in errors.items()
            }
        }
        
        # Guardar
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(error_stats, f, indent=2)
        
        print(f"✓ Análisis de errores guardado en {save_path}")
        
        # Imprimir resumen
        print("\n" + "="*70)
        print("ANÁLISIS DE ERRORES")
        print("="*70)
        for task, stats in error_stats['errors_by_task'].items():
            print(f"{task}:")
            print(f"  Errores: {stats['count']}/{total} ({stats['error_rate']:.2%})")
        print("="*70)
        
        return error_stats
    
    def plot_metrics_comparison(
        self,
        metrics: Dict,
        save_path: str = 'evaluation_results/metrics_comparison.png'
    ):
        """Grafica comparación de métricas por tarea."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        tasks = ['tile_type', 'rotation', 'meeple_presence', 'meeple_position']
        task_names = ['Tipo', 'Rotación', 'Meeple', 'Posición']
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        
        # Preparar datos
        data = {metric: [] for metric in metric_names}
        
        for task in tasks:
            for metric in metric_names:
                data[metric].append(metrics[task][metric])
        
        # Graficar
        x = np.arange(len(tasks))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metric_names):
            offset = width * (i - 1.5)
            ax.bar(x + offset, data[metric], width, label=metric.capitalize())
        
        ax.set_xlabel('Tarea')
        ax.set_ylabel('Score')
        ax.set_title('Comparación de Métricas por Tarea')
        ax.set_xticks(x)
        ax.set_xticklabels(task_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparación de métricas guardada en {save_path}")
        plt.close()


def evaluate_model(
    model_path: str,
    test_annotations: str,
    output_dir: str = 'evaluation_results'
):
    """
    Evalúa un modelo entrenado.
    
    Args:
        model_path: Ruta al modelo (.pth)
        test_annotations: Anotaciones del conjunto de test
        output_dir: Directorio para guardar resultados
    """
    print("\n" + "="*70)
    print("EVALUACIÓN DEL MODELO")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Cargar modelo
    from model import create_model
    
    print(f"\nCargando modelo desde {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print("✓ Modelo cargado")
    
    # Cargar dataset de test
    print(f"\nCargando dataset de test...")
    from torch.utils.data import DataLoader
    
    test_dataset = CarcassonneDataset(
        annotations_file=test_annotations,
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    print(f"✓ Dataset cargado: {len(test_dataset)} muestras")
    
    # Evaluar
    print("\nEvaluando modelo...")
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*70)
    print(f"\nAccuracy General: {metrics['overall_accuracy']:.4f}")
    print("\nMétricas por Tarea:")
    
    for task in ['tile_type', 'rotation', 'meeple_presence', 'meeple_position']:
        print(f"\n{task.upper()}:")
        for metric_name, value in metrics[task].items():
            if metric_name not in ['predictions', 'targets']:
                print(f"  {metric_name}: {value:.4f}")
    
    print("="*70)
    
    # Guardar métricas
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, 'metrics.json')
    
    # Preparar métricas para JSON (sin las predicciones)
    metrics_json = {
        'overall_accuracy': metrics['overall_accuracy']
    }
    for task in ['tile_type', 'rotation', 'meeple_presence', 'meeple_position']:
        metrics_json[task] = {
            k: v for k, v in metrics[task].items() 
            if k not in ['predictions', 'targets']
        }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"\n✓ Métricas guardadas en {metrics_file}")
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    evaluator.plot_confusion_matrices(metrics, output_dir)
    evaluator.plot_metrics_comparison(metrics, os.path.join(output_dir, 'metrics_comparison.png'))
    
    # Análisis de errores
    print("\nGenerando análisis de errores...")
    evaluator.generate_error_analysis(metrics, test_dataset, os.path.join(output_dir, 'error_analysis.json'))
    
    print("\n✓ Evaluación completada")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar modelo de clasificación')
    parser.add_argument('model_path', type=str, help='Ruta al modelo (.pth)')
    parser.add_argument('test_annotations', type=str, help='Anotaciones de test')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_annotations, args.output)
