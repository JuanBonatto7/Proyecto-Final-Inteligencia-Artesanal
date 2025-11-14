# Detector de Meeples - Approach OpenCV

Este proyecto detecta meeples azules y negros en losetas de Carcassonne usando visión computacional con OpenCV. **No requiere anotaciones manuales ni entrenamiento de IA**.

> **⚠️ Importante**: Este sistema usa algoritmos clásicos de visión computacional (Hough Transform, análisis de color HSV), NO machine learning. No hay "entrenamiento" - solo ajuste de parámetros según tus imágenes.

## 🚀 Características

- ✅ **Detección automática** de meeples circulares
- ✅ **Clasificación de colores** (azul/negro)
- ✅ **Detección de borde** de losetas
- ✅ **División en 9 posiciones** según cuadrícula 3x3
- ✅ **Sin anotaciones** - funciona out-of-the-box
- ✅ **Visualización** de resultados

## 📋 Requisitos

```bash
pip install opencv-python numpy matplotlib
```

## 🏗️ Estructura del Proyecto

```
Primera_Version_IA_Meeples/
├── src/
│   └── meeple_detector_cv.py    # Detector principal
├── tiles/                       # Imágenes de losetas base
├── real_test_images/            # Tus fotos reales de Carcassonne ⭐
├── test_images/                 # Imágenes de prueba simuladas
├── test_detector.py             # Probar con una imagen
├── test_real_images.py          # Probar con tus imágenes reales ⭐
├── process_all.py               # Procesar todas las imágenes
├── tune_params.py               # Ajustar parámetros
└── README.md
```

## 🎯 Cómo Usar

### ⭐ 1. Colocar Tus Imágenes Reales

**Pon tus fotos de Carcassonne en la carpeta `real_test_images/`:**
- Formatos soportados: JPG, PNG, BMP
- Incluye losetas con meeples azules y negros
- Fotos tomadas con buena iluminación
- Loseta centrada en la imagen

### ⭐ 2. Probar con Tus Imágenes

```bash
# Probar todas tus imágenes reales
python test_real_images.py
```

Esto procesará todas las imágenes en `real_test_images/` y:
- Mostrará resultados de detección
- Generará visualizaciones en `visualizations/`
- Guardará resultados en `real_test_results.json`

### 3. Probar con una Imagen Específica

```bash
python test_detector.py real_test_images/tu_foto.jpg
```

Esto mostrará:
- Si se detectó la loseta
- Cantidad de meeples encontrados
- Color y posición de cada meeple
- Visualización gráfica

### 3. Procesar Todas las Imágenes

```bash
python process_all.py
```

Esto procesará todas las imágenes en `tiles/` y generará:
- `detection_results.json`: Resultados detallados
- `visualizations/`: Imágenes con detecciones visualizadas
- Estadísticas completas

### ⭐ 4. Ajustar el Detector (Si es Necesario)

**Este NO es entrenamiento de IA - son ajustes de parámetros de visión computacional:**

```bash
# Ajustar parámetros de detección de círculos
python tune_params.py real_test_images/tu_foto.jpg --mode circles

# Ver y ajustar rangos de color
python tune_params.py real_test_images/tu_foto.jpg --mode colors
```

Los parámetros que puedes ajustar:
- **Detección de círculos**: Tamaño mínimo/máximo, sensibilidad
- **Colores**: Rangos HSV para azul y negro
- **Umbrales**: Sensibilidad de detección de bordes

## 🔧 Cómo Funciona

### 1. Detección de Bordes
- Usa Canny edge detection para encontrar contornos
- Identifica el contorno rectangular principal (la loseta)

### 2. División en 9 Zonas
```
0 1 2
3 4 5
6 7 8
```

### 3. Detección de Círculos
- Hough Circle Transform para encontrar formas circulares
- Parámetros ajustables para diferentes condiciones de iluminación

### 4. Clasificación de Color
- Conversión a espacio HSV
- Rangos predefinidos para azul y negro
- Análisis de histograma en la región del círculo

## 📊 Resultados

El sistema genera:
- **JSON con resultados**: Posición y color de cada meeple
- **Estadísticas**: Distribución por color y posición
- **Visualizaciones**: Imágenes marcadas con detecciones

## 🎮 Ejemplo de Salida

```
📊 RESULTADOS:
Loseta detectada: Sí
Meeples encontrados: 2

1. Meeple blue en posición 4
   Centro: (150, 200), Radio: 25

2. Meeple black en posición 7
   Centro: (300, 350), Radio: 22
```

## 🔍 Parámetros Ajustables

### Detección de Círculos
- `dp`: Resolución del acumulador
- `minDist`: Distancia mínima entre círculos
- `param1`, `param2`: Parámetros de Canny
- `minRadius`, `maxRadius`: Tamaño de círculos

### Rangos de Color (HSV)
```python
'blue': {'lower': [90, 50, 50], 'upper': [130, 255, 255]}
'black': {'lower': [0, 0, 0], 'upper': [180, 255, 30]}
```

## 🚀 Próximos Pasos

1. **Probar** con tus imágenes
2. **Ajustar parámetros** si es necesario
3. **Integrar** con tu sistema de juego de Carcassonne
4. **Optimizar** para diferentes condiciones de iluminación

¡El sistema está listo para usar sin necesidad de entrenamiento!