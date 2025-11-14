"""
Configuración central del clasificador de losetas de Carcassonne
"""

# Parámetros del modelo
IMG_SIZE = (224, 224)
NUM_CLASSES = 24
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 1e-5

# Rutas de directorios
DATASET_DIR = 'dataset/tiles/'
UNLABELED_DIR = 'dataset/unlabeled/'
MODELS_DIR = 'models/'
RESULTS_DIR = 'results/'

# Parámetros de Data Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 360,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'fill_mode': 'nearest',
    'brightness_range': [0.8, 1.2],
}

# Parámetros de entrenamiento
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Parámetros de generación sintética
SYNTHETIC_VARIATIONS_PER_IMAGE = 50

# Parámetros de clustering
N_CLUSTERS = 24
CLUSTERING_RANDOM_STATE = 42