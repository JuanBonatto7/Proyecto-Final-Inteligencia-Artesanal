"""
__init__.py - Paquete de Clasificación de Losetas de Carcassonne
"""

from .model import CarcassonneCNN, MultiTaskLoss, create_model
from .dataset import CarcassonneDataset, create_dataloaders, split_annotations
from .train import Trainer, train_model
from .inference import TileClassifier, classify_tiles_from_detector
from .evaluate import ModelEvaluator, evaluate_model
from .config import MODEL_CONFIG, TRAINING_CONFIG, LOSS_CONFIG

__version__ = '1.0.0'

__all__ = [
    'CarcassonneCNN',
    'MultiTaskLoss',
    'create_model',
    'CarcassonneDataset',
    'create_dataloaders',
    'split_annotations',
    'Trainer',
    'train_model',
    'TileClassifier',
    'classify_tiles_from_detector',
    'ModelEvaluator',
    'evaluate_model',
    'MODEL_CONFIG',
    'TRAINING_CONFIG',
    'LOSS_CONFIG'
]
