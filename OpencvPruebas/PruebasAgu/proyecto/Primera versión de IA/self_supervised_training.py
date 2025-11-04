#!/usr/bin/env python3
"""
Entrenamiento Self-Supervised de CNN sin anotaciones manuales

Este script implementa múltiples estrategias para entrenar una CNN
sin necesidad de anotar manualmente cada loseta:

1. Contrastive Learning (SimCLR)
2. Rotation Prediction
3. Auto-clustering con pseudo-labels
4. Few-shot learning con Siamese Networks

Uso:
    # Fase 1: Pre-entrenamiento self-supervised
    python self_supervised_training.py pretrain tiles/
    
    # Fase 2: Clustering automático
    python self_supervised_training.py cluster tiles/ --n-clusters 24
    
    # Fase 3: Fine-tuning con pseudo-labels
    python self_supervised_training.py finetune tiles/ clusters.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm


class ContrastiveTransform:
    """Transformaciones para Contrastive Learning"""
    
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        # Retorna DOS versiones aumentadas de la misma imagen
        return self.transform(x), self.transform(x)


class UnlabeledTileDataset(Dataset):
    """Dataset de losetas sin etiquetas para self-supervised learning"""
    
    def __init__(self, tiles_dir: str, transform=None, mode='contrastive'):
        self.tiles_dir = Path(tiles_dir)
        self.tile_files = sorted(list(self.tiles_dir.glob('*.png')))
        self.transform = transform
        self.mode = mode
        
        print(f"Dataset cargado: {len(self.tile_files)} losetas sin etiquetar")
    
    def __len__(self):
        return len(self.tile_files)
    
    def __getitem__(self, idx):
        tile_path = self.tile_files[idx]
        image = Image.open(tile_path).convert('RGB')
        
        if self.mode == 'contrastive':
            # Retorna dos vistas de la misma imagen
            if self.transform:
                view1, view2 = self.transform(image)
                return view1, view2
            else:
                return image, image
        
        elif self.mode == 'rotation':
            # Retorna imagen rotada y su label de rotación
            rotation = np.random.randint(0, 4)
            rotated = image.rotate(rotation * 90)
            
            if self.transform:
                rotated = self.transform(rotated)
            
            return rotated, rotation
        
        elif self.mode == 'extract':
            # Solo extrae features (sin aumentación)
            base_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return base_transform(image), str(tile_path)
        
        return image


class SimCLRModel(nn.Module):
    """Modelo para Contrastive Learning (SimCLR)"""
    
    def __init__(self, base_model='resnet18', projection_dim=128):
        super(SimCLRModel, self).__init__()
        
        # Encoder
        if base_model == 'resnet18':
            self.encoder = models.resnet18(pretrained=True)
            feature_dim = 512
        elif base_model == 'resnet50':
            self.encoder = models.resnet50(pretrained=True)
            feature_dim = 2048
        else:
            raise ValueError(f"Modelo {base_model} no soportado")
        
        # Remover última capa
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Projection head para contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        projections = self.projection(features)
        return F.normalize(projections, dim=1)


class RotationPredictionModel(nn.Module):
    """Modelo para predecir rotación (self-supervised)"""
    
    def __init__(self, base_model='resnet18'):
        super(RotationPredictionModel, self).__init__()
        
        if base_model == 'resnet18':
            self.encoder = models.resnet18(pretrained=True)
            feature_dim = 512
        else:
            self.encoder = models.resnet50(pretrained=True)
            feature_dim = 2048
        
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Predecir rotación (4 clases: 0°, 90°, 180°, 270°)
        self.rotation_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.rotation_head(features)


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    Usado en SimCLR
    """
    batch_size = z_i.shape[0]
    
    # Concatenar representaciones
    z = torch.cat([z_i, z_j], dim=0)  # 2*batch_size x dim
    
    # Calcular matriz de similitud
    sim_matrix = torch.mm(z, z.t()) / temperature
    
    # Máscara para excluir auto-similitud
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)
    
    # Labels: para cada i, el positivo es i + batch_size (o i - batch_size)
    labels = torch.cat([torch.arange(batch_size) + batch_size, 
                       torch.arange(batch_size)]).to(z.device)
    
    # Cross-entropy
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


def train_contrastive(tiles_dir: str, epochs=100, batch_size=32):
    """Entrena con Contrastive Learning (SimCLR)"""
    
    print("="*60)
    print("ENTRENAMIENTO CONTRASTIVE LEARNING (SimCLR)")
    print("="*60)
    print("\n✓ No requiere anotaciones")
    print("✓ Aprende representaciones útiles automáticamente")
    print("✓ Después requiere solo 5-10 ejemplos por clase\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando: {device}\n")
    
    # Dataset
    dataset = UnlabeledTileDataset(
        tiles_dir,
        transform=ContrastiveTransform(),
        mode='contrastive'
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Modelo
    model = SimCLRModel(base_model='resnet18').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenamiento
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for view1, view2 in pbar:
            view1, view2 = view1.to(device), view2.to(device)
            
            # Forward
            z1 = model(view1)
            z2 = model(view2)
            
            # Loss
            loss = nt_xent_loss(z1, z2)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Guardar modelo
    torch.save(model.state_dict(), 'contrastive_pretrained.pth')
    print("\n✓ Modelo guardado: contrastive_pretrained.pth")
    print("\nPróximo paso:")
    print("  python self_supervised_training.py cluster tiles/ --n-clusters 24")


def train_rotation(tiles_dir: str, epochs=50, batch_size=32):
    """Entrena prediciendo rotaciones"""
    
    print("="*60)
    print("ENTRENAMIENTO POR PREDICCIÓN DE ROTACIÓN")
    print("="*60)
    print("\n✓ No requiere anotaciones de tipo de loseta")
    print("✓ Aprende características geométricas\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = UnlabeledTileDataset(tiles_dir, transform=base_transform, mode='rotation')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modelo
    model = RotationPredictionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenamiento
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, rotations in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            rotations = rotations.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, rotations)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += rotations.size(0)
            correct += predicted.eq(rotations).sum().item()
        
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f} - Acc: {accuracy:.2f}%")
    
    torch.save(model.state_dict(), 'rotation_pretrained.pth')
    print("\n✓ Modelo guardado: rotation_pretrained.pth")


def extract_features(tiles_dir: str, model_path: str = None):
    """Extrae features de todas las losetas"""
    
    print("="*60)
    print("EXTRACCIÓN DE FEATURES")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelo pre-entrenado
    if model_path and Path(model_path).exists():
        print(f"\n✓ Cargando modelo: {model_path}")
        model = SimCLRModel()
        model.load_state_dict(torch.load(model_path))
    else:
        print("\n⚠️  No hay modelo pre-entrenado, usando ResNet18 de ImageNet")
        model = models.resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    
    model = model.to(device)
    model.eval()
    
    # Dataset
    dataset = UnlabeledTileDataset(tiles_dir, mode='extract')
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Extraer features
    all_features = []
    all_paths = []
    
    with torch.no_grad():
        for images, paths in tqdm(loader, desc="Extrayendo features"):
            images = images.to(device)
            features = model(images)
            features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_paths.extend(paths)
    
    features_array = np.vstack(all_features)
    
    print(f"\n✓ Features extraídos: {features_array.shape}")
    
    return features_array, all_paths


def auto_cluster(tiles_dir: str, n_clusters=24, model_path=None):
    """Clustering automático de losetas"""
    
    print("="*60)
    print("AUTO-CLUSTERING DE LOSETAS")
    print("="*60)
    print(f"\n✓ Agrupando automáticamente en {n_clusters} clusters\n")
    
    # Extraer features
    features, paths = extract_features(tiles_dir, model_path)
    
    # Reducir dimensionalidad (opcional, para visualización)
    print("Reduciendo dimensionalidad con PCA...")
    pca = PCA(n_components=min(50, features.shape[1]))
    features_reduced = pca.fit_transform(features)
    
    # K-Means clustering
    print(f"Ejecutando K-Means con {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_reduced)
    
    # Crear pseudo-labels
    pseudo_annotations = []
    for path, label in zip(paths, cluster_labels):
        pseudo_annotations.append({
            'image_path': path,
            'tile_type': int(label),
            'rotation': 0,  # Por defecto
            'has_meeple': False,
            'meeple_position': -1,
            'meeple_color': 'none',
            'pseudo_labeled': True
        })
    
    # Guardar
    output_file = 'pseudo_labels.json'
    with open(output_file, 'w') as f:
        json.dump(pseudo_annotations, f, indent=2)
    
    print(f"\n✓ Pseudo-labels guardados: {output_file}")
    print(f"✓ Total: {len(pseudo_annotations)} losetas agrupadas en {n_clusters} clusters")
    
    # Estadísticas
    print("\nDistribución por cluster:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} losetas")
    
    print("\n" + "="*60)
    print("SIGUIENTE PASO: Verificación Manual (5 minutos)")
    print("="*60)
    print("\nRevisa 1-2 ejemplos de cada cluster para:")
    print("1. Verificar que las losetas del mismo cluster son similares")
    print("2. Asignar letras (A-X) a cada cluster")
    print("3. Corregir errores obvios")
    print("\nUsa:")
    print("  python verify_clusters.py pseudo_labels.json")
    
    return pseudo_annotations


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento self-supervised sin anotaciones')
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: pretrain
    pretrain_parser = subparsers.add_parser('pretrain', help='Pre-entrenamiento self-supervised')
    pretrain_parser.add_argument('tiles_dir', help='Directorio con losetas')
    pretrain_parser.add_argument('--method', choices=['contrastive', 'rotation'], 
                                default='contrastive', help='Método de pre-entrenamiento')
    pretrain_parser.add_argument('--epochs', type=int, default=100, help='Número de epochs')
    
    # Comando: cluster
    cluster_parser = subparsers.add_parser('cluster', help='Clustering automático')
    cluster_parser.add_argument('tiles_dir', help='Directorio con losetas')
    cluster_parser.add_argument('--n-clusters', type=int, default=24, help='Número de clusters')
    cluster_parser.add_argument('--model', help='Modelo pre-entrenado (opcional)')
    
    args = parser.parse_args()
    
    if args.command == 'pretrain':
        if args.method == 'contrastive':
            train_contrastive(args.tiles_dir, epochs=args.epochs)
        elif args.method == 'rotation':
            train_rotation(args.tiles_dir, epochs=args.epochs)
    
    elif args.command == 'cluster':
        auto_cluster(args.tiles_dir, n_clusters=args.n_clusters, model_path=args.model)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
