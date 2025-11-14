"""
Sistema de Entrenamiento para el Clasificador de Losetas de Carcassonne

Este script maneja el entrenamiento completo del modelo, incluyendo:
- Early stopping
- Checkpoints
- Métricas detalladas
- Logging
- Visualización de progreso
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import CarcassonneCNN, MultiTaskLoss, create_model
from dataset import create_dataloaders, collate_fn


class Trainer:
    """Clase principal para entrenar el modelo."""
    
    def __init__(
        self,
        model: CarcassonneCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: MultiTaskLoss,
        optimizer: optim.Optimizer,
        device: torch.device,
        output_dir: str = 'models',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs'
    ):
        """
        Inicializa el trainer.
        
        Args:
            model: Modelo a entrenar
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación
            criterion: Función de pérdida
            optimizer: Optimizador
            device: Dispositivo (cuda/cpu)
            output_dir: Directorio para guardar modelos
            checkpoint_dir: Directorio para checkpoints
            log_dir: Directorio para logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # Directorios
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Historial de métricas
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_tile_acc': [],
            'val_tile_acc': [],
            'train_rotation_acc': [],
            'val_rotation_acc': [],
            'train_meeple_acc': [],
            'val_meeple_acc': [],
            'learning_rate': []
        }
        
        # Mejor modelo
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Entrena por una época.
        
        Returns:
            Tuple con (pérdida promedio, métricas)
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        # Contadores de aciertos
        correct = {
            'tile_type': 0,
            'rotation': 0,
            'meeple_presence': 0,
            'meeple_position': 0
        }
        
        # Barra de progreso
        pbar = tqdm(self.train_loader, desc='Training')
        
        for images, labels in pbar:
            # Mover a device
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calcular pérdida
            loss, loss_dict = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Actualizar métricas
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calcular accuracy
            for key in correct.keys():
                preds = torch.argmax(outputs[key], dim=1)
                correct[key] += (preds == labels[key]).sum().item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'tile_acc': f"{correct['tile_type'] / total_samples:.3f}",
                'rot_acc': f"{correct['rotation'] / total_samples:.3f}"
            })
        
        # Calcular métricas promedio
        avg_loss = total_loss / total_samples
        metrics = {
            'tile_type_acc': correct['tile_type'] / total_samples,
            'rotation_acc': correct['rotation'] / total_samples,
            'meeple_presence_acc': correct['meeple_presence'] / total_samples,
            'meeple_position_acc': correct['meeple_position'] / total_samples,
            'overall_acc': sum(correct.values()) / (total_samples * len(correct))
        }
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Valida el modelo.
        
        Returns:
            Tuple con (pérdida promedio, métricas)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        # Contadores de aciertos
        correct = {
            'tile_type': 0,
            'rotation': 0,
            'meeple_presence': 0,
            'meeple_position': 0
        }
        
        # Barra de progreso
        pbar = tqdm(self.val_loader, desc='Validation')
        
        for images, labels in pbar:
            # Mover a device
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass
            outputs = self.model(images)
            
            # Calcular pérdida
            loss, _ = self.criterion(outputs, labels)
            
            # Actualizar métricas
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calcular accuracy
            for key in correct.keys():
                preds = torch.argmax(outputs[key], dim=1)
                correct[key] += (preds == labels[key]).sum().item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'tile_acc': f"{correct['tile_type'] / total_samples:.3f}",
                'rot_acc': f"{correct['rotation'] / total_samples:.3f}"
            })
        
        # Calcular métricas promedio
        avg_loss = total_loss / total_samples
        metrics = {
            'tile_type_acc': correct['tile_type'] / total_samples,
            'rotation_acc': correct['rotation'] / total_samples,
            'meeple_presence_acc': correct['meeple_presence'] / total_samples,
            'meeple_position_acc': correct['meeple_position'] / total_samples,
            'overall_acc': sum(correct.values()) / (total_samples * len(correct))
        }
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Guarda un checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Guardar checkpoint regular
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Mejor modelo guardado en {best_path}")
        
        # Guardar último modelo
        last_path = self.output_dir / 'last_model.pth'
        torch.save(checkpoint, last_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Carga un checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch']
    
    def plot_history(self, save_path: Optional[str] = None):
        """Grafica el historial de entrenamiento."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Pérdida
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy general
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Overall Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy por tipo
        axes[1, 0].plot(self.history['train_tile_acc'], label='Train Tile')
        axes[1, 0].plot(self.history['val_tile_acc'], label='Val Tile')
        axes[1, 0].plot(self.history['train_rotation_acc'], label='Train Rotation')
        axes[1, 0].plot(self.history['val_rotation_acc'], label='Val Rotation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Tile Type and Rotation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfica guardada en {save_path}")
        else:
            plt.show()
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_every: int = 5
    ):
        """
        Loop principal de entrenamiento.
        
        Args:
            num_epochs: Número de épocas
            early_stopping_patience: Paciencia para early stopping
            scheduler: Scheduler de learning rate
            save_every: Guardar checkpoint cada N épocas
        """
        print("\n" + "="*70)
        print("INICIANDO ENTRENAMIENTO")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Épocas: {num_epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print("="*70 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nÉpoca {epoch}/{num_epochs}")
            print("-" * 70)
            
            # Entrenar
            train_loss, train_metrics = self.train_epoch()
            
            # Validar
            val_loss, val_metrics = self.validate()
            
            # Actualizar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_metrics['overall_acc'])
            self.history['val_acc'].append(val_metrics['overall_acc'])
            self.history['train_tile_acc'].append(train_metrics['tile_type_acc'])
            self.history['val_tile_acc'].append(val_metrics['tile_type_acc'])
            self.history['train_rotation_acc'].append(train_metrics['rotation_acc'])
            self.history['val_rotation_acc'].append(val_metrics['rotation_acc'])
            self.history['train_meeple_acc'].append(train_metrics['meeple_presence_acc'])
            self.history['val_meeple_acc'].append(val_metrics['meeple_presence_acc'])
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Imprimir métricas
            print(f"\nResultados Época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc:  {train_metrics['overall_acc']:.4f} | Val Acc:  {val_metrics['overall_acc']:.4f}")
            print(f"  Tile Acc:   {train_metrics['tile_type_acc']:.4f} | {val_metrics['tile_type_acc']:.4f}")
            print(f"  Rotation:   {train_metrics['rotation_acc']:.4f} | {val_metrics['rotation_acc']:.4f}")
            print(f"  Meeple:     {train_metrics['meeple_presence_acc']:.4f} | {val_metrics['meeple_presence_acc']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Verificar si es el mejor modelo
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_metrics['overall_acc']
                self.best_epoch = epoch
                self.patience_counter = 0
                print(f"  🎉 ¡Nuevo mejor modelo!")
            else:
                self.patience_counter += 1
                print(f"  Paciencia: {self.patience_counter}/{early_stopping_patience}")
            
            # Guardar checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⚠️ Early stopping activado en época {epoch}")
                print(f"   Mejor modelo: época {self.best_epoch} con val_loss={self.best_val_loss:.4f}")
                break
        
        # Tiempo total
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*70)
        print(f"Tiempo total: {total_time/60:.2f} minutos")
        print(f"Mejor época: {self.best_epoch}")
        print(f"Mejor val_loss: {self.best_val_loss:.4f}")
        print(f"Mejor val_acc: {self.best_val_acc:.4f}")
        print("="*70 + "\n")
        
        # Guardar historial
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Historial guardado en {history_path}")
        
        # Graficar
        plot_path = self.log_dir / 'training_curves.png'
        self.plot_history(save_path=plot_path)


def train_model(
    train_annotations: str,
    val_annotations: str,
    config: Optional[Dict] = None,
    resume_from: Optional[str] = None
):
    """
    Función principal para entrenar el modelo.
    
    Args:
        train_annotations: Ruta al archivo de anotaciones de entrenamiento
        val_annotations: Ruta al archivo de anotaciones de validación
        config: Configuración de entrenamiento
        resume_from: Ruta a checkpoint para continuar entrenamiento
    """
    # Configuración por defecto
    if config is None:
        config = {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'image_size': 224,
            'num_workers': 4,
            'early_stopping_patience': 15,
            'save_every': 5,
            'backbone': 'efficientnet_b0',
            'dropout': 0.3
        }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Crear dataloaders
    print("\nCargando datos...")
    train_loader, val_loader = create_dataloaders(
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=config['num_workers']
    )
    
    # Crear modelo
    print("\nCreando modelo...")
    model = create_model({
        'backbone': config['backbone'],
        'dropout': config['dropout'],
        'pretrained': True
    })
    
    # Criterio y optimizador
    criterion = MultiTaskLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Crear trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # Reanudar si es necesario
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nReanudando desde {resume_from}")
        start_epoch = trainer.load_checkpoint(resume_from)
    
    # Entrenar
    trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        scheduler=scheduler,
        save_every=config['save_every']
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar clasificador de losetas')
    parser.add_argument('--train', type=str, required=True, help='Archivo de anotaciones de entrenamiento')
    parser.add_argument('--val', type=str, required=True, help='Archivo de anotaciones de validación')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', 
                        choices=['efficientnet_b0', 'resnet18', 'resnet34', 'resnet50'],
                        help='Backbone de la CNN')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint para reanudar')
    
    args = parser.parse_args()
    
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'backbone': args.backbone,
        'weight_decay': 1e-4,
        'image_size': 224,
        'num_workers': 4,
        'early_stopping_patience': 15,
        'save_every': 5,
        'dropout': 0.3
    }
    
    train_model(
        train_annotations=args.train,
        val_annotations=args.val,
        config=config,
        resume_from=args.resume
    )
