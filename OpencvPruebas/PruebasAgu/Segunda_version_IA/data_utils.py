"""
Utilidades para manejo y generación de datos
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from pathlib import Path
import numpy as np
import shutil
from config import (
    IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, AUGMENTATION_CONFIG,
    SYNTHETIC_VARIATIONS_PER_IMAGE
)
from model import create_data_augmentation


def create_data_generators(dataset_dir):
    """
    Crea generadores de datos para entrenamiento y validación
    
    Args:
        dataset_dir: Directorio con las carpetas de clases
    
    Returns:
        train_generator, validation_generator
    """
    train_datagen = ImageDataGenerator(
        rescaling=1./255,
        validation_split=VALIDATION_SPLIT,
        **AUGMENTATION_CONFIG
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator


def generate_synthetic_variations(image_path, output_dir, num_variations=SYNTHETIC_VARIATIONS_PER_IMAGE):
    """
    Genera variaciones sintéticas de una imagen
    
    Args:
        image_path: Ruta de la imagen original
        output_dir: Directorio donde guardar las variaciones
        num_variations: Número de variaciones a generar
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    
    augmenter = create_data_augmentation()
    
    # Guardar imagen original
    img.save(output_dir / f"{Path(image_path).stem}_original.jpg")
    
    # Generar y guardar variaciones
    for i in range(num_variations):
        augmented = augmenter(tf.expand_dims(img_array, 0), training=True)
        augmented_img = tf.keras.preprocessing.image.array_to_img(augmented[0])
        augmented_img.save(output_dir / f"{Path(image_path).stem}_var_{i:03d}.jpg")
    
    print(f"Generadas {num_variations} variaciones de {image_path}")


def batch_generate_synthetic_data(input_dir, output_base_dir, num_variations=SYNTHETIC_VARIATIONS_PER_IMAGE):
    """
    Genera variaciones sintéticas para todas las imágenes en un directorio
    Mantiene la estructura de carpetas (clases)
    
    Args:
        input_dir: Directorio con carpetas de clases
        output_base_dir: Directorio base de salida
        num_variations: Variaciones por imagen
    """
    input_path = Path(input_dir)
    output_path = Path(output_base_dir)
    
    # Procesar cada carpeta de clase
    for class_dir in input_path.iterdir():
        if class_dir.is_dir():
            print(f"\nProcesando clase: {class_dir.name}")
            output_class_dir = output_path / class_dir.name
            
            # Procesar cada imagen en la clase
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            for img_file in image_files:
                generate_synthetic_variations(
                    img_file,
                    output_class_dir,
                    num_variations
                )
    
    print(f"\n✓ Dataset sintético generado en: {output_base_dir}")


def extract_features_from_images(image_paths, model):
    """
    Extrae características de imágenes usando un modelo
    
    Args:
        image_paths: Lista de rutas a imágenes
        model: Modelo para extracción de características
    
    Returns:
        Array numpy con características
    """
    features = []
    
    for img_path in image_paths:
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        
        feature = model.predict(img_array, verbose=0)
        features.append(feature[0])
    
    return np.array(features)


def get_class_distribution(dataset_dir):
    """
    Muestra la distribución de imágenes por clase
    
    Args:
        dataset_dir: Directorio con carpetas de clases
    
    Returns:
        Diccionario con conteo por clase
    """
    distribution = {}
    
    for class_dir in Path(dataset_dir).iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
            distribution[class_dir.name] = count
    
    return distribution


def print_dataset_info(dataset_dir):
    """
    Imprime información del dataset
    """
    distribution = get_class_distribution(dataset_dir)
    
    print("\n" + "="*50)
    print("INFORMACIÓN DEL DATASET")
    print("="*50)
    print(f"Número de clases: {len(distribution)}")
    print(f"Total de imágenes: {sum(distribution.values())}")
    print(f"\nDistribución por clase:")
    print("-"*50)
    
    for class_name, count in sorted(distribution.items()):
        print(f"  {class_name:30s}: {count:4d} imágenes")
    
    print("="*50 + "\n")