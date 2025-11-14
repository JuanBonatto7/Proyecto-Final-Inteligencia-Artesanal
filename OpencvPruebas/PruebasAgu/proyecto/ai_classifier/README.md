# 🎮 Sistema de IA para Clasificación de Losetas de Carcassonne

Sistema completo de clasificación basado en Deep Learning para identificar y clasificar las 72 losetas de un tablero de Carcassonne.

## 📋 Características

- **Multi-tarea**: Clasifica simultáneamente:
  - Tipo de loseta (A-X + BLANCO, 25 clases)
  - Rotación (0-3, 4 orientaciones)
  - Presencia de meeple (Sí/No)
  - Posición del meeple (0-8, 9 posiciones)

- **Transfer Learning**: Usa modelos preentrenados (EfficientNet/ResNet)
- **Data Augmentation**: Aumenta automáticamente los datos de entrenamiento
- **Easy to Train**: Interface simple y clara para entrenar
- **Herramienta de Anotación**: Interface gráfica para etiquetar datos
- **Análisis Completo**: Métricas detalladas y visualizaciones

## 🚀 Instalación

### Requisitos

- Python 3.8+
- PyTorch 1.12+
- CUDA (opcional, para GPU)

### Instalar dependencias

```bash
pip install torch torchvision
pip install opencv-python pillow
pip install numpy scikit-learn matplotlib seaborn
pip install tqdm
```

O usando el archivo de requirements:

```bash
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
ai_classifier/
│
├── model.py              # Arquitectura de la CNN multi-tarea
├── dataset.py            # Dataset y DataLoader
├── train.py              # Sistema de entrenamiento
├── inference.py          # Pipeline de inferencia/predicción
├── annotate.py           # Herramienta de anotación interactiva
├── evaluate.py           # Evaluación y métricas
├── config.py             # Configuraciones
│
├── models/               # Modelos entrenados
├── checkpoints/          # Checkpoints de entrenamiento
├── logs/                 # Logs y visualizaciones
└── evaluation_results/   # Resultados de evaluación
```

## 🎯 Workflow Completo

### 1️⃣ Preparar Datos

#### Paso 1: Extraer losetas con el detector

```bash
cd "Reconocimiento de losetas con 8 referencias"
python carcassonne.py foto_tablero.jpg
# Esto creará una carpeta tiles/ con las 72 losetas individuales
```

#### Paso 2: Anotar las losetas

```bash
cd ai_classifier
python annotate.py ../tiles --output annotations.json
```

**Controles de anotación:**
- `A-Z`: Seleccionar tipo de loseta
- `B`: Loseta BLANCO
- `0-3`: Rotación (0=0°, 1=90°, 2=180°, 3=270°)
- `M`: Toggle presencia de meeple
- `0-8`: Posición del meeple (si hay)
- `ENTER`: Guardar y siguiente
- `←/→`: Navegar
- `S`: Guardar progreso
- `ESC`: Salir

#### Paso 3: Dividir en train/val

```python
from dataset import split_annotations

train_file, val_file = split_annotations(
    annotations_file='annotations.json',
    train_ratio=0.8,
    output_dir='data'
)
```

### 2️⃣ Entrenar el Modelo

#### Entrenamiento básico

```bash
python train.py --train data/train_annotations.json --val data/val_annotations.json
```

#### Entrenamiento avanzado

```bash
python train.py \
    --train data/train_annotations.json \
    --val data/val_annotations.json \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --backbone efficientnet_b0
```

**Opciones disponibles:**
- `--epochs`: Número de épocas (default: 100)
- `--batch-size`: Tamaño del batch (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--backbone`: Arquitectura base (`efficientnet_b0`, `resnet18`, `resnet34`, `resnet50`)
- `--resume`: Checkpoint para continuar entrenamiento

#### Monitorear entrenamiento

El sistema guarda automáticamente:
- `models/best_model.pth`: Mejor modelo
- `models/last_model.pth`: Último modelo
- `checkpoints/checkpoint_epoch_*.pth`: Checkpoints periódicos
- `logs/training_history.json`: Historial completo
- `logs/training_curves.png`: Gráficas de entrenamiento

### 3️⃣ Evaluar el Modelo

```bash
python evaluate.py models/best_model.pth data/test_annotations.json
```

Genera:
- `evaluation_results/metrics.json`: Métricas detalladas
- `evaluation_results/confusion_matrices.png`: Matrices de confusión
- `evaluation_results/metrics_comparison.png`: Comparación de métricas
- `evaluation_results/error_analysis.json`: Análisis de errores

### 4️⃣ Hacer Predicciones

#### Clasificar una sola loseta

```bash
python inference.py models/best_model.pth single tile.png --visualize
```

#### Clasificar un directorio completo

```bash
python inference.py models/best_model.pth batch tiles/ --output predictions.json
```

#### Integrar con el detector

```python
from inference import classify_tiles_from_detector

classify_tiles_from_detector(
    detector_tiles_dir='tiles/',
    model_path='models/best_model.pth',
    output_json='results.json'
)
```

## 💻 Uso Programático

### Entrenar desde Python

```python
from train import train_model

config = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'backbone': 'efficientnet_b0',
    'dropout': 0.3
}

train_model(
    train_annotations='data/train_annotations.json',
    val_annotations='data/val_annotations.json',
    config=config
)
```

### Hacer predicciones

```python
from inference import TileClassifier

# Cargar modelo
classifier = TileClassifier('models/best_model.pth')

# Predecir
result = classifier.predict_single('tile.png')

print(f"Tipo: {result['tile_letter']}")
print(f"Rotación: {result['rotation']} ({result['rotation'] * 90}°)")
print(f"Meeple: {'Sí' if result['has_meeple'] else 'No'}")
if result['has_meeple']:
    print(f"Posición: {result['meeple_position']}")
```

### Evaluar

```python
from evaluate import evaluate_model

evaluate_model(
    model_path='models/best_model.pth',
    test_annotations='data/test_annotations.json',
    output_dir='evaluation_results'
)
```

## 🎨 Arquitectura del Modelo

```
Input (224x224x3)
    ↓
EfficientNet-B0 Backbone (Pretrained)
    ↓
Shared FC Layer (512 neurons)
    ↓
    ├─→ Tile Type Head → 25 classes (A-X + BLANCO)
    ├─→ Rotation Head → 4 classes (0-3)
    ├─→ Meeple Presence Head → 2 classes (Sí/No)
    └─→ Meeple Position Head → 9 classes (0-8)
```

## 📊 Métricas de Rendimiento

El sistema reporta:
- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: Calidad de las predicciones positivas
- **Recall**: Cobertura de casos positivos
- **F1-Score**: Balance entre precision y recall
- **Matrices de confusión**: Errores por clase
- **Análisis de errores**: Top errores y patrones

## 🔧 Configuración Avanzada

### Modificar arquitectura

Edita `config.py`:

```python
MODEL_CONFIG = {
    'backbone': 'resnet50',  # Cambiar backbone
    'dropout': 0.4,          # Aumentar dropout
    'pretrained': True
}
```

### Ajustar data augmentation

Edita las transformaciones en `dataset.py`:

```python
transforms.RandomRotation(degrees=15),  # Más rotación
transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Más variación de color
```

### Pesos de pérdida

Si un tipo de clasificación es más importante:

```python
LOSS_CONFIG = {
    'tile_type_weight': 3.0,  # Más peso al tipo
    'rotation_weight': 1.0,
    'meeple_presence_weight': 1.5,
    'meeple_position_weight': 1.0
}
```

## 🐛 Troubleshooting

### Problema: Modelo no converge

**Solución:**
- Reducir learning rate: `--lr 0.0001`
- Aumentar batch size: `--batch-size 64`
- Usar más data augmentation
- Verificar que las anotaciones sean correctas

### Problema: Overfitting

**Solución:**
- Aumentar dropout: editar `MODEL_CONFIG['dropout'] = 0.5`
- Más data augmentation
- Early stopping (ya incluido)
- Reducir tamaño del modelo: usar `resnet18` en vez de `resnet50`

### Problema: Underfitting

**Solución:**
- Aumentar capacidad del modelo: usar `resnet50` o `efficientnet_b3`
- Entrenar más épocas
- Reducir regularización (dropout)
- Verificar calidad de datos

### Problema: GPU out of memory

**Solución:**
- Reducir batch size: `--batch-size 16`
- Reducir image size en `config.py`: `image_size = 128`
- Usar modelo más pequeño: `resnet18`

## 📚 Tips para Mejorar el Modelo

1. **Más datos**: Anotar más tableros diferentes
2. **Balance de clases**: Asegurar representación uniforme de tipos
3. **Data augmentation**: Experimentar con diferentes aumentaciones
4. **Ensemble**: Combinar predicciones de múltiples modelos
5. **Fine-tuning**: Ajustar capas del backbone con learning rate bajo

## 🤝 Integración con el Sistema de Detección

Script completo de ejemplo:

```python
import os
from pathlib import Path

# 1. Detectar losetas
os.system('python carcassonne.py tablero.jpg')

# 2. Clasificar losetas
from inference import classify_tiles_from_detector

results = classify_tiles_from_detector(
    detector_tiles_dir='tiles/',
    model_path='../ai_classifier/models/best_model.pth',
    output_json='board_classification.json'
)

# 3. Procesar resultados
print(f"✓ {len(results)} losetas clasificadas")
print(f"✓ Resultados en board_classification.json")
```

## 📝 Formato de Anotaciones

```json
[
  {
    "image_path": "tiles/tile_001.png",
    "tile_letter": "A",
    "tile_type": 0,
    "rotation": 2,
    "has_meeple": true,
    "meeple_position": 4,
    "confidence": 1.0,
    "auto_annotated": false
  }
]
```

## 🎓 Próximos Pasos

1. **Active Learning**: Sistema que sugiere qué imágenes anotar
2. **Semi-supervised**: Aprovechar datos no anotados
3. **Detección de meeples**: CNN específica para detectar posición
4. **Modelo ligero**: Versión optimizada para dispositivos móviles
5. **API REST**: Servicio web para clasificación

## 📄 Licencia

Este proyecto es parte del Proyecto Final de Inteligencia Artesanal.

## 👥 Autores

Proyecto Carcassonne - Universidad

---

**¡Buena suerte con tu entrenamiento! 🚀**
