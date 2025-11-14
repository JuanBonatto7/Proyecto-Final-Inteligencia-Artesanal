"""
Script para realizar predicciones sobre nuevas imágenes
"""

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from config import IMG_SIZE, MODELS_DIR


def load_model_and_mapping(model_path, mapping_path=None):
    """
    Carga el modelo y el mapeo de clases
    
    Args:
        model_path: Ruta al archivo .h5 del modelo
        mapping_path: Ruta al JSON con mapeo de clases (opcional)
    
    Returns:
        model, class_mapping
    """
    print(f"📦 Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Cargar mapeo de clases
    class_mapping = None
    if mapping_path:
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
            # Convertir keys a int
            class_mapping = {int(k): v for k, v in class_mapping.items()}
    else:
        # Buscar archivo de mapeo automáticamente
        model_name = Path(model_path).stem.replace('_final', '').replace('_best', '')
        possible_mapping = Path(MODELS_DIR) / f"{model_name}_class_mapping.json"
        
        if possible_mapping.exists():
            with open(possible_mapping, 'r') as f:
                class_mapping = json.load(f)
                class_mapping = {int(k): v for k, v in class_mapping.items()}
            print(f"✓ Mapeo de clases cargado desde: {possible_mapping}")
    
    print(f"✓ Modelo cargado con {len(model.layers)} capas")
    
    return model, class_mapping


def preprocess_image(image_path):
    """
    Preprocesa una imagen para predicción
    
    Args:
        image_path: Ruta a la imagen
    
    Returns:
        Array numpy listo para predicción
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_tile(model, image_path, class_mapping=None, top_k=3):
    """
    Predice el tipo de loseta de una imagen
    
    Args:
        model: Modelo de Keras cargado
        image_path: Ruta a la imagen
        class_mapping: Diccionario de mapeo de clases
        top_k: Número de predicciones principales a retornar
    
    Returns:
        dict con predicciones y probabilidades
    """
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Obtener top-k predicciones
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = {
        'image_path': str(image_path),
        'predictions': []
    }
    
    for idx in top_indices:
        class_name = class_mapping[idx] if class_mapping else f"Clase_{idx}"
        confidence = float(predictions[idx])
        
        results['predictions'].append({
            'class_id': int(idx),
            'class_name': class_name,
            'confidence': confidence
        })
    
    return results


def batch_predict(model, image_dir, class_mapping=None, top_k=3):
    """
    Realiza predicciones sobre todas las imágenes en un directorio
    
    Args:
        model: Modelo de Keras cargado
        image_dir: Directorio con imágenes
        class_mapping: Diccionario de mapeo de clases
        top_k: Número de predicciones principales a retornar
    
    Returns:
        Lista de resultados de predicción
    """
    image_path = Path(image_dir)
    image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
    
    print(f"\n🔍 Procesando {len(image_files)} imágenes...")
    
    all_results = []
    for img_file in image_files:
        result = predict_tile(model, img_file, class_mapping, top_k)
        all_results.append(result)
        
        # Mostrar resultado
        top_pred = result['predictions'][0]
        print(f"  {img_file.name:40s} → {top_pred['class_name']:20s} ({top_pred['confidence']:.2%})")
    
    return all_results


def visualize_prediction(model, image_path, class_mapping=None, save_path=None):
    """
    Visualiza una predicción con la imagen y las probabilidades
    
    Args:
        model: Modelo de Keras cargado
        image_path: Ruta a la imagen
        class_mapping: Diccionario de mapeo de clases
        save_path: Ruta donde guardar la visualización (opcional)
    """
    results = predict_tile(model, image_path, class_mapping, top_k=5)
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Mostrar imagen
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Imagen: {Path(image_path).name}', fontsize=12, fontweight='bold')
    
    # Mostrar probabilidades
    classes = [p['class_name'] for p in results['predictions']]
    confidences = [p['confidence'] for p in results['predictions']]
    
    colors = ['green' if i == 0 else 'lightblue' for i in range(len(classes))]
    
    ax2.barh(classes, confidences, color=colors)
    ax2.set_xlabel('Probabilidad', fontsize=11)
    ax2.set_title('Top 5 Predicciones', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    # Añadir porcentajes
    for i, (cls, conf) in enumerate(zip(classes, confidences)):
        ax2.text(conf + 0.02, i, f'{conf:.1%}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualización guardada en: {save_path}")
    else:
        plt.show()
    
    plt.close()


def predict_from_board(model, board_tiles_dir, class_mapping=None, output_json=None):
    """
    Predice todas las losetas individualizadas de un tablero
    
    Args:
        model: Modelo de Keras cargado
        board_tiles_dir: Directorio con losetas individualizadas
        class_mapping: Diccionario de mapeo de clases
        output_json: Ruta para guardar resultados en JSON (opcional)
    
    Returns:
        Diccionario con predicciones organizadas
    """
    results = batch_predict(model, board_tiles_dir, class_mapping)
    
    # Organizar resultados
    board_prediction = {
        'total_tiles': len(results),
        'tiles': results,
        'summary': {}
    }
    
    # Contar predicciones por clase
    for result in results:
        top_class = result['predictions'][0]['class_name']
        board_prediction['summary'][top_class] = board_prediction['summary'].get(top_class, 0) + 1
    
    # Guardar JSON si se especifica
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(board_prediction, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Resultados guardados en: {output_json}")
    
    # Mostrar resumen
    print("\n📊 Resumen de predicciones:")
    print("="*60)
    for class_name, count in sorted(board_prediction['summary'].items()):
        print(f"  {class_name:30s}: {count:3d} losetas")
    print("="*60 + "\n")
    
    return board_prediction


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python predict.py <modelo.h5> <imagen_o_directorio>")
        print("Ejemplo: python predict.py models/carcassonne_classifier_final.h5 test_images/")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    
    # Cargar modelo
    model, class_mapping = load_model_and_mapping(model_path)
    
    # Predecir
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Predicción individual
        print("\n🎯 Predicción individual:")
        results = predict_tile(model, input_path, class_mapping)
        
        for i, pred in enumerate(results['predictions'], 1):
            print(f"{i}. {pred['class_name']}: {pred['confidence']:.2%}")
        
        # Visualizar
        visualize_prediction(model, input_path, class_mapping)
        
    elif input_path.is_dir():
        # Predicción en lote
        results = predict_from_board(
            model,
            input_path,
            class_mapping,
            output_json='predictions_output.json'
        )
    else:
        print(f"❌ Error: {input_path} no existe")