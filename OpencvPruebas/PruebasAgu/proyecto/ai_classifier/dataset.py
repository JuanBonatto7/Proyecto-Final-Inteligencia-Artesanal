"""
Dataset y DataLoader para Losetas de Carcassonne

Este módulo maneja la carga de datos, preprocesamiento y data augmentation.
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CarcassonneDataset(Dataset):
    """
    Dataset personalizado para losetas de Carcassonne.
    
    Lee imágenes de losetas y sus anotaciones (tipo, rotación, meeple).
    """
    
    # Mapeo de letras a índices
    TILE_LETTERS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'BLANCO'
    ]
    
    LETTER_TO_IDX = {letter: idx for idx, letter in enumerate(TILE_LETTERS)}
    IDX_TO_LETTER = {idx: letter for idx, letter in enumerate(TILE_LETTERS)}
    
    def __init__(
        self,
        annotations_file: str,
        root_dir: str = None,
        transform: transforms.Compose = None,
        image_size: int = 224,
        augment: bool = False,
        normalize: bool = True
    ):
        """
        Inicializa el dataset.
        
        Args:
            annotations_file: Ruta al archivo JSON con anotaciones
            root_dir: Directorio raíz de las imágenes (si no está en annotations)
            transform: Transformaciones personalizadas
            image_size: Tamaño al que redimensionar las imágenes
            augment: Si aplicar data augmentation
            normalize: Si normalizar las imágenes
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        # Cargar anotaciones
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Filtrar anotaciones válidas
        self.samples = []
        for ann in self.annotations:
            if self._is_valid_annotation(ann):
                self.samples.append(ann)
        
        print(f"Dataset cargado: {len(self.samples)} muestras válidas de {len(self.annotations)} totales")
        
        # Crear transformaciones
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._create_transforms()
    
    def _is_valid_annotation(self, annotation: Dict) -> bool:
        """Verifica que una anotación sea válida."""
        required_keys = ['image_path', 'tile_letter', 'rotation', 'has_meeple', 'meeple_position']
        
        # Verificar que tenga todas las claves necesarias
        if not all(key in annotation for key in required_keys):
            return False
        
        # Verificar que la imagen exista
        image_path = self._get_full_image_path(annotation['image_path'])
        if not os.path.exists(image_path):
            return False
        
        # Verificar valores válidos
        if annotation['tile_letter'] not in self.LETTER_TO_IDX:
            return False
        
        if annotation['rotation'] not in [0, 1, 2, 3]:
            return False
        
        if annotation['has_meeple'] not in [True, False]:
            return False
        
        if annotation['meeple_position'] not in list(range(9)) + [-1]:
            return False
        
        # Verificar meeple_color si existe (compatibilidad con anotaciones antiguas)
        if 'meeple_color' in annotation:
            if annotation['meeple_color'] not in ['blue', 'black', None]:
                return False
        
        return True
    
    def _get_full_image_path(self, image_path: str) -> str:
        """Obtiene la ruta completa de la imagen."""
        if self.root_dir is not None:
            return os.path.join(self.root_dir, image_path)
        return image_path
    
    def _create_transforms(self) -> transforms.Compose:
        """Crea las transformaciones apropiadas."""
        transform_list = []
        
        # Redimensionar
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Data augmentation (solo si está habilitado)
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            ])
        
        # Convertir a tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalizar (usando estadísticas de ImageNet)
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Obtiene un elemento del dataset.
        
        Returns:
            - Imagen como tensor (3, H, W)
            - Diccionario con etiquetas:
                - 'tile_type': Índice del tipo de loseta
                - 'rotation': Rotación (0-3)
                - 'meeple_presence': 0 o 1
                - 'meeple_position': 0-8 o -1
                - 'meeple_color': 0=blue, 1=black, o -1 si no hay meeple
        """
        annotation = self.samples[idx]
        
        # Cargar imagen
        image_path = self._get_full_image_path(annotation['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Aplicar transformaciones
        image = self.transform(image)
        
        # Preparar etiquetas
        tile_type = self.LETTER_TO_IDX[annotation['tile_letter']]
        rotation = annotation['rotation']
        meeple_presence = 1 if annotation['has_meeple'] else 0
        meeple_position = annotation['meeple_position'] if annotation['has_meeple'] else -1
        
        # Meeple color: 0=blue, 1=black, -1=sin meeple
        meeple_color = -1
        if annotation['has_meeple'] and 'meeple_color' in annotation:
            if annotation['meeple_color'] == 'blue':
                meeple_color = 0
            elif annotation['meeple_color'] == 'black':
                meeple_color = 1
        
        labels = {
            'tile_type': tile_type,
            'rotation': rotation,
            'meeple_presence': meeple_presence,
            'meeple_position': meeple_position,
            'meeple_color': meeple_color
        }
        
        return image, labels
    
    def get_class_distribution(self) -> Dict[str, Dict]:
        """Calcula la distribución de clases en el dataset."""
        tile_types = {}
        rotations = {0: 0, 1: 0, 2: 0, 3: 0}
        meeples = {0: 0, 1: 0}
        positions = {i: 0 for i in range(-1, 9)}
        
        for sample in self.samples:
            # Tipo de loseta
            tile_letter = sample['tile_letter']
            tile_types[tile_letter] = tile_types.get(tile_letter, 0) + 1
            
            # Rotación
            rotations[sample['rotation']] += 1
            
            # Meeple
            has_meeple = 1 if sample['has_meeple'] else 0
            meeples[has_meeple] += 1
            
            # Posición
            pos = sample['meeple_position'] if sample['has_meeple'] else -1
            positions[pos] += 1
        
        return {
            'tile_types': tile_types,
            'rotations': rotations,
            'meeples': meeples,
            'positions': positions
        }


def create_dataloaders(
    train_annotations: str,
    val_annotations: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    root_dir: str = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders para entrenamiento y validación.
    
    Args:
        train_annotations: Ruta al archivo de anotaciones de entrenamiento
        val_annotations: Ruta al archivo de anotaciones de validación
        batch_size: Tamaño del batch
        image_size: Tamaño de las imágenes
        num_workers: Número de workers para cargar datos
        root_dir: Directorio raíz de las imágenes
        
    Returns:
        Tuple con (train_loader, val_loader)
    """
    # Dataset de entrenamiento (con augmentation)
    train_dataset = CarcassonneDataset(
        annotations_file=train_annotations,
        root_dir=root_dir,
        image_size=image_size,
        augment=True,
        normalize=True
    )
    
    # Dataset de validación (sin augmentation)
    val_dataset = CarcassonneDataset(
        annotations_file=val_annotations,
        root_dir=root_dir,
        image_size=image_size,
        augment=False,
        normalize=True
    )
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Función personalizada para agrupar muestras en un batch.
    
    Args:
        batch: Lista de tuplas (imagen, etiquetas)
        
    Returns:
        Tuple con (imágenes_batch, etiquetas_batch)
    """
    images = []
    tile_types = []
    rotations = []
    meeple_presences = []
    meeple_positions = []
    meeple_colors = []
    
    for image, labels in batch:
        images.append(image)
        tile_types.append(labels['tile_type'])
        rotations.append(labels['rotation'])
        meeple_presences.append(labels['meeple_presence'])
        meeple_positions.append(labels['meeple_position'])
        meeple_colors.append(labels['meeple_color'])
    
    images = torch.stack(images)
    labels = {
        'tile_type': torch.tensor(tile_types, dtype=torch.long),
        'rotation': torch.tensor(rotations, dtype=torch.long),
        'meeple_presence': torch.tensor(meeple_presences, dtype=torch.long),
        'meeple_position': torch.tensor(meeple_positions, dtype=torch.long),
        'meeple_color': torch.tensor(meeple_colors, dtype=torch.long)
    }
    
    return images, labels


def split_annotations(
    annotations_file: str,
    train_ratio: float = 0.8,
    output_dir: str = None,
    random_seed: int = 42
) -> Tuple[str, str]:
    """
    Divide un archivo de anotaciones en train y validation.
    
    Args:
        annotations_file: Ruta al archivo de anotaciones completo
        train_ratio: Proporción de datos para entrenamiento
        output_dir: Directorio donde guardar los archivos divididos
        random_seed: Semilla para reproducibilidad
        
    Returns:
        Tuple con (train_file, val_file)
    """
    # Cargar anotaciones
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Mezclar
    random.seed(random_seed)
    random.shuffle(annotations)
    
    # Dividir
    split_idx = int(len(annotations) * train_ratio)
    train_annotations = annotations[:split_idx]
    val_annotations = annotations[split_idx:]
    
    # Determinar directorio de salida
    if output_dir is None:
        output_dir = os.path.dirname(annotations_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar archivos
    train_file = os.path.join(output_dir, 'train_annotations.json')
    val_file = os.path.join(output_dir, 'val_annotations.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_annotations, f, indent=2)
    
    print(f"Anotaciones divididas:")
    print(f"  Train: {len(train_annotations)} muestras → {train_file}")
    print(f"  Val: {len(val_annotations)} muestras → {val_file}")
    
    return train_file, val_file


if __name__ == "__main__":
    # Test del dataset
    print("=== Test del Dataset ===\n")
    
    # Crear un dataset de ejemplo
    # (Asumiendo que existe un archivo de anotaciones)
    test_annotations = "test_annotations.json"
    
    if os.path.exists(test_annotations):
        dataset = CarcassonneDataset(
            annotations_file=test_annotations,
            augment=True
        )
        
        print(f"Dataset: {len(dataset)} muestras")
        
        # Obtener una muestra
        image, labels = dataset[0]
        print(f"\nMuestra 0:")
        print(f"  Imagen shape: {image.shape}")
        print(f"  Labels: {labels}")
        
        # Distribución de clases
        distribution = dataset.get_class_distribution()
        print(f"\nDistribución de clases:")
        print(f"  Tipos de loseta: {distribution['tile_types']}")
        print(f"  Rotaciones: {distribution['rotations']}")
        print(f"  Meeples: {distribution['meeples']}")
    else:
        print(f"No se encontró el archivo {test_annotations}")
        print("Crea anotaciones usando la herramienta de anotación.")
