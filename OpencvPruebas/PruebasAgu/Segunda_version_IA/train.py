"""
Script principal de entrenamiento del clasificador
"""

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import json
from datetime import datetime

from config import (
    DATASET_DIR, MODELS_DIR, RESULTS_DIR, EPOCHS,
    LEARNING_RATE, FINE_TUNE_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR
)
from model import create_classification_model, compile_model, unfreeze_top_layers
from data_utils import create_data_generators, print_dataset_info


def create_callbacks(model_name):
    """
    Crea los callbacks para el entrenamiento
    
    Args:
        model_name: Nombre del modelo para guardar checkpoints
    
    Returns:
        Lista de callbacks
    """
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{MODELS_DIR}/{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            f"{RESULTS_DIR}/{model_name}_training_log.csv"
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path):
    """
    Grafica el historial de entrenamiento
    
    Args:
        history: Objeto History de Keras
        save_path: Ruta donde guardar el gráfico
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Entrenamiento')
    axes[0].plot(history.history['val_accuracy'], label='Validación')
    axes[0].set_title('Precisión del Modelo')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Entrenamiento')
    axes[1].plot(history.history['val_loss'], label='Validación')
    axes[1].set_title('Pérdida del Modelo')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfico de entrenamiento guardado en: {save_path}")


def train_model(dataset_dir=DATASET_DIR, epochs=EPOCHS, fine_tune=True):
    """
    Entrena el modelo de clasificación
    
    Args:
        dataset_dir: Directorio con el dataset organizado
        epochs: Número de épocas de entrenamiento
        fine_tune: Si True, realiza fine-tuning después del entrenamiento inicial
    
    Returns:
        Modelo entrenado
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"carcassonne_classifier_{timestamp}"
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO DEL CLASIFICADOR DE LOSETAS DE CARCASSONNE")
    print("="*70)
    
    # Información del dataset
    print_dataset_info(dataset_dir)
    
    # Crear generadores de datos
    print("🔧 Preparando generadores de datos...")
    train_gen, val_gen = create_data_generators(dataset_dir)
    
    num_classes = len(train_gen.class_indices)
    print(f"✓ Clases detectadas: {num_classes}")
    print(f"✓ Imágenes de entrenamiento: {train_gen.samples}")
    print(f"✓ Imágenes de validación: {val_gen.samples}")
    
    # Guardar mapeo de clases
    class_mapping = {v: k for k, v in train_gen.class_indices.items()}
    with open(f"{MODELS_DIR}/{model_name}_class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"✓ Mapeo de clases guardado")
    
    # Crear modelo
    print("\n🏗️  Creando modelo...")
    model = create_classification_model(num_classes=num_classes, trainable_base=False)
    model = compile_model(model, LEARNING_RATE)
    
    model.summary()
    
    # Entrenamiento inicial
    print("\n🚀 Iniciando entrenamiento (fase 1: transfer learning)...")
    print("="*70)
    
    callbacks = create_callbacks(f"{model_name}_phase1")
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Fase 1 completada")
    
    # Guardar resultados de fase 1
    plot_training_history(
        history1,
        f"{RESULTS_DIR}/{model_name}_phase1_history.png"
    )
    
    # Fine-tuning (opcional)
    if fine_tune:
        print("\n🔥 Iniciando fine-tuning (fase 2)...")
        print("="*70)
        
        model = unfreeze_top_layers(model, num_layers=30)
        model = compile_model(model, FINE_TUNE_LEARNING_RATE)
        
        callbacks = create_callbacks(f"{model_name}_phase2")
        
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs // 2,  # Menos épocas para fine-tuning
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Fase 2 completada")
        
        # Guardar resultados de fase 2
        plot_training_history(
            history2,
            f"{RESULTS_DIR}/{model_name}_phase2_history.png"
        )
    
    # Guardar modelo final
    final_model_path = f"{MODELS_DIR}/{model_name}_final.h5"
    model.save(final_model_path)
    print(f"\n✓ Modelo final guardado en: {final_model_path}")
    
    # Evaluación final
    print("\n📊 Evaluación final en conjunto de validación:")
    print("="*70)
    results = model.evaluate(val_gen, verbose=0)
    print(f"Loss: {results[0]:.4f}")
    print(f"Accuracy: {results[1]:.4f}")
    print(f"Top-K Accuracy: {results[2]:.4f}")
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70 + "\n")
    
    return model, model_name


if __name__ == "__main__":
    # Verificar GPU
    print("🖥️  Dispositivos disponibles:")
    print(tf.config.list_physical_devices())
    
    # Entrenar
    model, model_name = train_model()
    
    print(f"\n💾 Archivos generados:")
    print(f"  - Modelo: {MODELS_DIR}/{model_name}_final.h5")
    print(f"  - Mapeo de clases: {MODELS_DIR}/{model_name}_class_mapping.json")
    print(f"  - Gráficos: {RESULTS_DIR}/{model_name}_*.png")
    print(f"  - Logs: {RESULTS_DIR}/{model_name}_training_log.csv")