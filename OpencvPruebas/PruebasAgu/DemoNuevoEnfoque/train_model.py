#!/usr/bin/env python3
"""
Script de entrenamiento simplificado para Carcassonne CNN
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from carcassonne_cnn import (
    CarcassonneTileDataset, 
    CarcassonneCNN, 
    CarcassonneTrainer,
    create_data_transforms
)

def train(train_file, val_file, epochs=50, batch_size=32):
    print("="*60)
    print("ENTRENAMIENTO DEL MODELO CARCASSONNE")
    print("="*60)
    
    # Verificar archivos
    if not Path(train_file).exists():
        print(f"✗ Error: No se encuentra {train_file}")
        return False
    
    if not Path(val_file).exists():
        print(f"✗ Error: No se encuentra {val_file}")
        return False
    
    print(f"\n✓ Archivos de datos encontrados")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    
    # Cargar datasets
    print(f"\nCargando datasets...")
    train_dataset = CarcassonneTileDataset(
        train_file,
        transform=create_data_transforms(augment=True)
    )
    val_dataset = CarcassonneTileDataset(
        val_file,
        transform=create_data_transforms(augment=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Crear modelo
    print(f"\nCreando modelo...")
    model = CarcassonneCNN()
    
    # Entrenar
    trainer = CarcassonneTrainer(model)
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Graficar historia
    trainer.plot_history()
    
    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print(f"Modelo guardado en: best_carcassonne_model.pth")
    print(f"Gráfica guardada en: training_history.png")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python train_model.py <train_annotations.json> <val_annotations.json> [epochs]")
        print("\nEjemplo:")
        print("  python train_model.py train_annotations.json val_annotations.json 50")
        sys.exit(1)
    
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    train(train_file, val_file, epochs)
