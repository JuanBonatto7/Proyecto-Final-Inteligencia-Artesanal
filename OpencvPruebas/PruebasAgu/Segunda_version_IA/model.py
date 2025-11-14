"""
Definición del modelo de clasificación de losetas
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from config import IMG_SIZE, NUM_CLASSES


def create_data_augmentation():
    """Crea el pipeline de data augmentation"""
    return tf.keras.Sequential([
        layers.RandomRotation(1.0),  # Rotación completa 360°
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
        layers.RandomZoom(0.2),
    ])


def create_classification_model(num_classes=NUM_CLASSES, trainable_base=False):
    """
    Crea el modelo de clasificación usando transfer learning
    
    Args:
        num_classes: Número de clases (tipos de losetas)
        trainable_base: Si True, permite entrenar las capas del modelo base
    
    Returns:
        Modelo compilado de Keras
    """
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base
    
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_contrastive_model():
    """
    Crea modelo para self-supervised learning (opcional)
    Útil para aprender representaciones sin etiquetas
    
    Returns:
        Modelo de embeddings
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    embeddings = layers.Dense(128)(x)
    
    return models.Model(inputs, embeddings)


def compile_model(model, learning_rate):
    """
    Compila el modelo con configuración estándar
    
    Args:
        model: Modelo de Keras a compilar
        learning_rate: Tasa de aprendizaje
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    return model


def unfreeze_top_layers(model, num_layers=30):
    """
    Descongela las últimas capas del modelo base para fine-tuning
    
    Args:
        model: Modelo a modificar
        num_layers: Número de capas superiores a descongelar
    """
    # Encontrar la capa base (MobileNetV2)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    
    if base_model:
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        print(f"Descongeladas las últimas {num_layers} capas del modelo base")
    
    return model