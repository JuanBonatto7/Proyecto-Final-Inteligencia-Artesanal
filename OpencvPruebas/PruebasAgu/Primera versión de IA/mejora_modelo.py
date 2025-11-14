#!/usr/bin/env python3
"""
Estrategias para mejorar un modelo CNN que no está funcionando perfectamente

Este script implementa múltiples técnicas para mejorar el rendimiento:
1. Continuar entrenamiento (más epochs)
2. Fine-tuning con ajuste de learning rate
3. Data augmentation más agresivo
4. Corrección de errores y re-entrenamiento
5. Transfer learning avanzado
6. Ensemble de modelos

Uso:
    # Continuar entrenamiento
    python mejora_modelo.py continue best_carcassonne_model.pth train.json val.json
    
    # Fine-tuning con learning rate bajo
    python mejora_modelo.py finetune best_carcassonne_model.pth train.json val.json
    
    # Re-entrenar con más datos
    python mejora_modelo.py retrain train.json val.json --augment-factor 20
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import sys
import argparse

from carcassonne_cnn import (
    CarcassonneTileDataset,
    CarcassonneCNN,
    CarcassonneTrainer,
    create_data_transforms
)


def continue_training(model_path: str, train_file: str, val_file: str, 
                     additional_epochs: int = 50, new_lr: float = None):
    """
    Continúa el entrenamiento de un modelo existente
    
    Args:
        model_path: Ruta al modelo guardado
        train_file: Archivo de anotaciones de entrenamiento
        val_file: Archivo de anotaciones de validación
        additional_epochs: Epochs adicionales a entrenar
        new_lr: Nuevo learning rate (opcional, reduce para fine-tuning)
    """
    
    print("="*60)
    print("CONTINUAR ENTRENAMIENTO")
    print("="*60)
    
    # Verificar archivos
    if not Path(model_path).exists():
        print(f"✗ Error: Modelo no encontrado: {model_path}")
        return False
    
    print(f"\n✓ Cargando modelo desde: {model_path}")
    
    # Cargar datasets
    print("Cargando datasets...")
    train_dataset = CarcassonneTileDataset(
        train_file,
        transform=create_data_transforms(augment=True)
    )
    val_dataset = CarcassonneTileDataset(
        val_file,
        transform=create_data_transforms(augment=False)
    )
    
    print(f"✓ Train: {len(train_dataset)} muestras")
    print(f"✓ Val: {len(val_dataset)} muestras")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CarcassonneCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    print(f"✓ Modelo cargado en: {device}")
    
    # Crear trainer
    trainer = CarcassonneTrainer(model, device=device)
    
    # Ajustar learning rate si se especifica
    if new_lr is not None:
        print(f"\n✓ Ajustando learning rate a: {new_lr}")
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    # Continuar entrenamiento
    print(f"\nEntrenando {additional_epochs} epochs adicionales...")
    trainer.train(train_loader, val_loader, epochs=additional_epochs)
    
    # Graficar historia
    trainer.plot_history()
    
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO CONTINUADO COMPLETADO")
    print("="*60)
    print("Modelo mejorado guardado en: best_carcassonne_model.pth")
    
    return True


def finetune_model(model_path: str, train_file: str, val_file: str,
                   epochs: int = 30, lr: float = 0.0001):
    """
    Fine-tuning con learning rate muy bajo
    Útil cuando el modelo ya está cerca del óptimo
    """
    
    print("="*60)
    print("FINE-TUNING DEL MODELO")
    print("="*60)
    print(f"\nLearning rate: {lr} (muy bajo para ajuste fino)")
    
    return continue_training(model_path, train_file, val_file, 
                           additional_epochs=epochs, new_lr=lr)


def retrain_with_more_data(train_file: str, val_file: str, 
                          augment_factor: int = 20,
                          epochs: int = 100):
    """
    Re-entrena desde cero con data augmentation más agresivo
    """
    
    print("="*60)
    print("RE-ENTRENAMIENTO CON MÁS DATOS")
    print("="*60)
    
    print(f"\n⚠️  Esto re-entrenará desde cero")
    print(f"✓ Factor de augmentación: {augment_factor}x")
    print(f"✓ Epochs: {epochs}")
    
    # Primero generar datos aumentados
    print("\nGenerando datos aumentados...")
    import subprocess
    
    # Aumentar train
    cmd = f"python data-augmentation.py augment {train_file} data/augmented_retrain/ {augment_factor}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Error aumentando datos: {result.stderr}")
        return False
    
    # Entrenar con datos aumentados
    augmented_train = "data/augmented_retrain/augmented_annotations.json"
    
    if not Path(augmented_train).exists():
        print(f"✗ Error: No se generaron datos aumentados")
        return False
    
    print(f"\n✓ Datos aumentados generados")
    
    # Entrenar desde cero
    from train_model import train
    
    return train(augmented_train, val_file, epochs=epochs)


def correct_errors_and_retrain(model_path: str, train_file: str, val_file: str,
                               test_file: str = None):
    """
    Estrategia iterativa:
    1. Evaluar modelo
    2. Identificar errores
    3. Corregir/anotar casos problemáticos
    4. Re-entrenar
    """
    
    print("="*60)
    print("CORRECCIÓN ITERATIVA")
    print("="*60)
    
    print("\nEste proceso te ayudará a:")
    print("1. Identificar qué losetas clasifica mal")
    print("2. Anotar correctamente esas losetas")
    print("3. Re-entrenar con los datos corregidos")
    
    # Evaluar modelo actual
    if test_file and Path(test_file).exists():
        print("\nEvaluando modelo actual...")
        import subprocess
        result = subprocess.run(
            f"python model-evaluation.py {model_path} {test_file}",
            shell=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    
    print("\n" + "="*60)
    print("PASOS SIGUIENTES")
    print("="*60)
    
    print("\n1. Revisa los errores en el reporte de evaluación")
    print("2. Identifica patrones (¿qué tipos confunde más?)")
    print("3. Opciones:")
    print("\n   A) Anotar más ejemplos de los tipos problemáticos:")
    print("      python annotation_tool_letters.py tiles/ referencias/")
    print("\n   B) Mejorar las referencias de esos tipos")
    print("\n   C) Usar auto-anotación con threshold más bajo:")
    print("      python auto_annotate.py tiles/ referencias/ --threshold 0.55")
    print("\n4. Combinar con datos existentes:")
    print("   python combine_annotations.py train_annotations.json nuevas_anotaciones.json train_mejorado.json")
    print("\n5. Re-entrenar:")
    print("   python train_model.py train_mejorado.json val_annotations.json")
    
    return True


def progressive_unfreezing(model_path: str, train_file: str, val_file: str):
    """
    Descongelamiento progresivo de capas
    Útil cuando usas transfer learning
    """
    
    print("="*60)
    print("DESCONGELAMIENTO PROGRESIVO")
    print("="*60)
    
    print("\nEsta técnica:")
    print("1. Empieza entrenando solo las últimas capas")
    print("2. Gradualmente descongela capas anteriores")
    print("3. Permite fine-tuning más cuidadoso")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo
    model = CarcassonneCNN()
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Cargar datos
    train_dataset = CarcassonneTileDataset(
        train_file,
        transform=create_data_transforms(augment=True)
    )
    val_dataset = CarcassonneTileDataset(
        val_file,
        transform=create_data_transforms(augment=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Fase 1: Solo últimas capas
    print("\n[Fase 1/3] Entrenando solo las cabezas de clasificación...")
    
    # Congelar backbone
    for param in model.features.parameters():
        param.requires_grad = False
    
    trainer = CarcassonneTrainer(model, device=device)
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = 0.001
    
    trainer.train(train_loader, val_loader, epochs=20)
    
    # Fase 2: Descongelar últimas capas del backbone
    print("\n[Fase 2/3] Descongelando últimas capas del backbone...")
    
    # Descongelar últimas 3 capas
    for param in list(model.features.parameters())[-6:]:
        param.requires_grad = True
    
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = 0.0001
    
    trainer.train(train_loader, val_loader, epochs=20)
    
    # Fase 3: Todo descongelado con LR muy bajo
    print("\n[Fase 3/3] Fine-tuning completo...")
    
    for param in model.features.parameters():
        param.requires_grad = True
    
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = 0.00001
    
    trainer.train(train_loader, val_loader, epochs=20)
    
    trainer.plot_history()
    
    print("\n✓ Descongelamiento progresivo completado")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Mejorar modelo CNN de Carcassonne',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Estrategias disponibles:

1. CONTINUAR (continue):
   Continúa entrenando el modelo existente con más epochs
   python mejora_modelo.py continue best_carcassonne_model.pth train.json val.json --epochs 50

2. FINE-TUNING (finetune):
   Ajuste fino con learning rate muy bajo
   python mejora_modelo.py finetune best_carcassonne_model.pth train.json val.json --lr 0.0001

3. RE-ENTRENAR (retrain):
   Entrena desde cero con más data augmentation
   python mejora_modelo.py retrain train.json val.json --augment 20

4. CORRECCIÓN ITERATIVA (correct):
   Identifica errores y guía para corregirlos
   python mejora_modelo.py correct best_carcassonne_model.pth train.json val.json

5. DESCONGELAMIENTO PROGRESIVO (unfreeze):
   Fine-tuning cuidadoso capa por capa
   python mejora_modelo.py unfreeze best_carcassonne_model.pth train.json val.json
        """
    )
    
    parser.add_argument('strategy', 
                       choices=['continue', 'finetune', 'retrain', 'correct', 'unfreeze'],
                       help='Estrategia de mejora')
    parser.add_argument('model_or_train', help='Modelo (.pth) o archivo de train')
    parser.add_argument('train_or_val', help='Archivo de train o val')
    parser.add_argument('val', nargs='?', help='Archivo de validación')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs adicionales')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--augment', type=int, default=20, help='Factor de augmentación')
    parser.add_argument('--test', help='Archivo de test (opcional)')
    
    args = parser.parse_args()
    
    # Determinar archivos según estrategia
    if args.strategy in ['continue', 'finetune', 'unfreeze', 'correct']:
        model_path = args.model_or_train
        train_file = args.train_or_val
        val_file = args.val
        
        if not val_file:
            print("✗ Error: Necesitas especificar archivo de validación")
            sys.exit(1)
    else:  # retrain
        train_file = args.model_or_train
        val_file = args.train_or_val
    
    # Ejecutar estrategia
    if args.strategy == 'continue':
        success = continue_training(model_path, train_file, val_file, 
                                   additional_epochs=args.epochs)
    
    elif args.strategy == 'finetune':
        success = finetune_model(model_path, train_file, val_file,
                               epochs=args.epochs, lr=args.lr)
    
    elif args.strategy == 'retrain':
        success = retrain_with_more_data(train_file, val_file,
                                        augment_factor=args.augment,
                                        epochs=args.epochs)
    
    elif args.strategy == 'correct':
        success = correct_errors_and_retrain(model_path, train_file, val_file,
                                            test_file=args.test)
    
    elif args.strategy == 'unfreeze':
        success = progressive_unfreezing(model_path, train_file, val_file)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
