# Detector de Meeples Azules/Negros en Carcassonne

Este proyecto implementa una Inteligencia Artificial basada en CNN para detectar si una loseta de Carcassonne tiene un meeple azul o negro, y en qué posición se encuentra.

## Estructura del Proyecto

```
Primera_Version_IA_Meeples/
├── data/
│   ├── tiles/                 # Imágenes de losetas
│   ├── train_annotations.json # Anotaciones de entrenamiento
│   ├── val_annotations.json   # Anotaciones de validación
│   └── annotations_template.json # Plantilla para anotaciones
├── models/                    # Modelos entrenados
├── output/                    # Resultados y gráficos
├── src/                       # Código fuente
│   ├── meeple_detector.py     # Modelo y clases principales
│   ├── train_meeple_detector.py # Script de entrenamiento
│   └── predict_meeple.py      # Script de predicción
├── utils/                     # Utilidades
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Preparar Datos

- Coloca las imágenes de las losetas en `data/tiles/`
- Usa la plantilla `data/annotations_template.json` para crear anotaciones
- Crea `data/train_annotations.json` y `data/val_annotations.json`

### 2. Entrenar el Modelo

```bash
python src/train_meeple_detector.py
```

### 3. Hacer Predicciones

Para una imagen individual:
```bash
python src/predict_meeple.py ruta/a/la/imagen.png
```

Para un directorio con múltiples imágenes:
```bash
python src/predict_meeple.py data/tiles/
```

## Formato de Anotaciones

Cada anotación en el JSON debe tener:

```json
{
  "image_path": "data/tiles/tile_001.png",
  "has_blue_or_black_meeple": true,
  "meeple_position": 4,
  "meeple_color": "blue"
}
```

- `has_blue_or_black_meeple`: `true` si tiene meeple azul o negro
- `meeple_position`: posición 0-8 según la cuadrícula 3x3, -1 si no hay meeple
- `meeple_color`: `"blue"` o `"black"` (opcional)

### Numeración de Posiciones

La loseta se divide en 9 subespacios numerados así:

```
0 1 2
3 4 5
6 7 8
```

## Arquitectura del Modelo

- **Backbone**: ResNet18 pre-entrenado
- **Tareas**:
  - Detección de presencia de meeple (binaria)
  - Clasificación de posición (9 clases + no meeple)

## Resultados

Los resultados se guardan en `output/`:
- `best_meeple_model.pth`: Mejor modelo entrenado
- `training_history.png`: Gráfico del entrenamiento
- `prediction_results.json`: Resultados de predicción