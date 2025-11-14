# 📦 Sistema de IA para Clasificación de Losetas - Información del Proyecto

## 🎯 Descripción

Sistema completo de Inteligencia Artificial basado en Deep Learning para clasificar automáticamente las 72 losetas de un tablero de Carcassonne. El sistema identifica:

- **Tipo de loseta**: 25 clases (A-X + BLANCO)
- **Rotación**: 4 orientaciones (0°, 90°, 180°, 270°)
- **Presencia de meeple**: Sí/No
- **Posición del meeple**: 9 posiciones posibles (0-8)

## 🏗️ Arquitectura

### Modelo CNN Multi-tarea

```
Input (224x224x3)
    ↓
EfficientNet-B0 Backbone (Pretrained ImageNet)
    ↓
Shared Features (512 neurons)
    ↓
    ├─→ Tile Type Head    → 25 classes
    ├─→ Rotation Head     → 4 classes
    ├─→ Meeple Head       → 2 classes
    └─→ Position Head     → 9 classes
```

### Características Técnicas

- **Framework**: PyTorch
- **Backbone**: EfficientNet-B0 / ResNet (18/34/50)
- **Transfer Learning**: Sí (ImageNet)
- **Loss Function**: Multi-task weighted CrossEntropy
- **Optimizer**: AdamW con weight decay
- **Scheduler**: ReduceLROnPlateau
- **Data Augmentation**: Sí (rotation, flip, color jitter, affine)
- **Early Stopping**: Sí (patience=15)

## 📂 Estructura de Archivos

```
ai_classifier/
│
├── 🧠 Core Modules
│   ├── model.py              # Arquitectura CNN multi-tarea
│   ├── dataset.py            # Dataset y DataLoader
│   ├── train.py              # Sistema de entrenamiento
│   ├── inference.py          # Predicción e inferencia
│   ├── evaluate.py           # Evaluación y métricas
│   └── config.py             # Configuraciones
│
├── 🛠️ Herramientas
│   ├── annotate.py           # Herramienta de anotación GUI
│   ├── quick_start.py        # Script de inicio rápido
│   └── examples.py           # Ejemplos de uso
│
├── 📚 Documentación
│   ├── README.md             # Documentación completa
│   ├── QUICKSTART.md         # Guía rápida 5 minutos
│   └── INFO.md               # Este archivo
│
├── 📦 Dependencias
│   ├── requirements.txt      # Dependencias Python
│   └── __init__.py          # Inicializador del paquete
│
└── 📁 Directorios de salida
    ├── models/               # Modelos entrenados (.pth)
    ├── checkpoints/          # Checkpoints de entrenamiento
    ├── logs/                 # Logs y visualizaciones
    ├── data/                 # Datos y anotaciones
    └── evaluation_results/   # Resultados de evaluación
```

## 🚀 Comandos Principales

### Anotación de Datos
```bash
python annotate.py tiles/ --output annotations.json
```

### Entrenamiento
```bash
python train.py --train data/train_annotations.json --val data/val_annotations.json --epochs 100
```

### Evaluación
```bash
python evaluate.py models/best_model.pth data/test_annotations.json
```

### Inferencia
```bash
# Una imagen
python inference.py models/best_model.pth single tile.png --visualize

# Directorio completo
python inference.py models/best_model.pth batch tiles/ --output predictions.json
```

### Inicio Rápido
```bash
python quick_start.py
```

## 📊 Métricas y Visualizaciones

### Durante Entrenamiento

- Loss por época (train/val)
- Accuracy global
- Accuracy por tarea (tipo, rotación, meeple)
- Learning rate schedule
- Gráficas en tiempo real

### Después de Evaluación

- Accuracy, Precision, Recall, F1 por tarea
- Matrices de confusión
- Análisis de errores
- Ejemplos de predicciones incorrectas

## 🎮 Integración con el Detector

```python
# 1. Detector extrae losetas
os.system('python carcassonne.py tablero.jpg')

# 2. IA clasifica losetas
from inference import classify_tiles_from_detector

results = classify_tiles_from_detector(
    detector_tiles_dir='tiles/',
    model_path='models/best_model.pth',
    output_json='board_results.json'
)

# 3. Procesar resultados
for tile in results:
    print(f"{tile['tile_letter']} - Rot: {tile['rotation']}")
```

## 💾 Formato de Datos

### Anotaciones (JSON)

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

### Predicciones (JSON)

```json
{
  "image_path": "tiles/tile_001.png",
  "tile_type": 0,
  "tile_letter": "A",
  "rotation": 2,
  "has_meeple": true,
  "meeple_position": 4,
  "confidence": {
    "tile_type": 0.95,
    "rotation": 0.92,
    "meeple_presence": 0.88,
    "meeple_position": 0.76
  }
}
```

## 🔬 Características Avanzadas

### Transfer Learning
- Usa pesos preentrenados de ImageNet
- Fine-tuning de todo el modelo
- Convergencia más rápida y mejores resultados

### Data Augmentation
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation)
- Random affine transformations
- Random perspective

### Multi-task Learning
- Entrena todas las tareas simultáneamente
- Comparte características entre tareas
- Pesos ajustables por tarea

### Regularización
- Dropout (0.3)
- Weight decay (1e-4)
- Label smoothing (0.1)
- Early stopping

## 📈 Rendimiento Esperado

Con 300-500 losetas anotadas de calidad:

- **Tipo de loseta**: 85-95% accuracy
- **Rotación**: 90-98% accuracy
- **Meeple presencia**: 92-98% accuracy
- **Meeple posición**: 75-85% accuracy
- **Overall (todo correcto)**: 70-85% accuracy

## 🛠️ Requisitos del Sistema

### Mínimo
- Python 3.8+
- 4 GB RAM
- CPU multi-core
- 2 GB espacio en disco

### Recomendado
- Python 3.10+
- 8 GB RAM
- NVIDIA GPU (4+ GB VRAM)
- CUDA 11.0+
- 5 GB espacio en disco

## 🎓 Tecnologías Utilizadas

- **PyTorch**: Framework de deep learning
- **torchvision**: Modelos preentrenados y transformaciones
- **OpenCV**: Procesamiento de imágenes
- **PIL**: Carga de imágenes
- **NumPy**: Operaciones numéricas
- **scikit-learn**: Métricas y evaluación
- **Matplotlib/Seaborn**: Visualizaciones
- **tqdm**: Barras de progreso

## 📝 Licencia y Créditos

Proyecto Final - Inteligencia Artesanal
Universidad - 2025

## 🔮 Próximas Mejoras

- [ ] Active Learning para anotación eficiente
- [ ] Semi-supervised learning
- [ ] Detección automática de posición de meeple
- [ ] Modelo ligero para móviles
- [ ] API REST
- [ ] Interface web
- [ ] Exportación a ONNX/TensorFlow Lite
- [ ] Ensemble de modelos

## 📞 Soporte

Para problemas o preguntas:
1. Revisa README.md completo
2. Consulta QUICKSTART.md
3. Ejecuta ejemplos: `python examples.py`
4. Revisa issues en el repositorio

## 🎉 ¡Gracias por usar este sistema!

Esperamos que te sea útil para tu proyecto de Carcassonne.
