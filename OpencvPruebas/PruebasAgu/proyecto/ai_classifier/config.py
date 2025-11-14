"""
Configuración del Sistema de IA para Carcassonne
"""

# Configuración del modelo
MODEL_CONFIG = {
    'num_tile_types': 25,  # A-X (24) + BLANCO (1)
    'num_rotations': 4,     # 0, 90, 180, 270 grados
    'num_meeple_classes': 2,  # Con/Sin meeple
    'num_meeple_positions': 9,  # Posiciones 0-8
    'num_meeple_colors': 2,  # blue, black
    'backbone': 'efficientnet_b0',  # 'efficientnet_b0', 'resnet18', 'resnet34', 'resnet50'
    'pretrained': True,
    'dropout': 0.3
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'image_size': 224,
    'num_workers': 4,
    'early_stopping_patience': 15,
    'save_every': 5
}

# Configuración de pérdida
LOSS_CONFIG = {
    'tile_type_weight': 2.0,
    'rotation_weight': 1.0,
    'meeple_presence_weight': 1.5,
    'meeple_position_weight': 1.0,
    'meeple_color_weight': 1.0,
    'use_label_smoothing': True,
    'label_smoothing': 0.1
}

# Configuración de data augmentation
AUGMENTATION_CONFIG = {
    'random_horizontal_flip': 0.3,
    'random_rotation_degrees': 10,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    },
    'random_affine': {
        'degrees': 0,
        'translate': (0.1, 0.1),
        'scale': (0.9, 1.1)
    },
    'random_perspective': {
        'distortion_scale': 0.2,
        'p': 0.3
    }
}

# Directorios
DIRS = {
    'models': 'models',
    'checkpoints': 'checkpoints',
    'logs': 'logs',
    'data': 'data',
    'evaluation': 'evaluation_results'
}

# Tipos de losetas
TILE_TYPES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'BLANCO'
]
