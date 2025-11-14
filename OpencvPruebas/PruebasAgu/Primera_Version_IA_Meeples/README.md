# Detector de Meeples - Approach OpenCV

Este proyecto detecta **meeples azules y negros** en losetas de Carcassonne usando visión computacional con OpenCV. **No requiere anotaciones manuales ni entrenamiento de IA**.

> **✅ CORREGIDO**: El detector ahora funciona correctamente con meeples reales de Carcassonne (solo azul y negro, 1 por imagen).

# 🧠 Sistema Integrado de Detección de Meeples

Este proyecto combina **visión computacional clásica (OpenCV)** y **redes convolucionales (CNN)** para detectar meeples azules y negros en losetas de Carcassonne. Incluye herramientas para **anotación manual**, **evaluación automática** y **mejora iterativa** del detector.

> **🎯 NUEVO**: Sistema integrado con anotación manual, evaluación automática y opción de CNN
> **✅ ACTUALIZADO**: Rangos HSV precisos basados en valores reales (Azul: HSV(212,64%,62%), Negro: HSV(240,10%,8%))

## 🚀 Características

- ✅ **Anotación Manual Interactiva** - Marca meeples manualmente para crear ground truth
- ✅ **Evaluación Automática** - Compara detector vs anotaciones manuales
- ✅ **Detector OpenCV Mejorado** - Rangos HSV precisos, lógica optimizada
- ✅ **Opción CNN** - Entrenamiento de red convolucional con PyTorch
- ✅ **Sistema Integrado** - Interfaz unificada para todas las herramientas
- ✅ **Visualizaciones** - Comparaciones lado a lado
- ✅ **Métricas de Rendimiento** - Precisión, Recall, F1-Score

## 📋 Requisitos

```bash
pip install opencv-python numpy matplotlib torch torchvision scikit-learn seaborn
```

## 🏗️ Estructura del Proyecto

```
Primera_Version_IA_Meeples/
├── src/
│   └── meeple_detector_cv.py        # Detector OpenCV mejorado
├── real_test_images/                # Tus fotos reales ⭐
├── meeple_system.py                 # 🆕 Sistema integrado principal
├── meeple_annotator.py              # 🆕 Anotación manual interactiva
├── evaluate_detector.py             # 🆕 Evaluación automática
├── cnn_meeple_detector.py           # 🆕 Detector CNN con PyTorch
├── test_real_images.py              # Prueba batch con imágenes reales
├── manual_annotations.json          # 🆕 Anotaciones ground truth
├── evaluation_results.json          # 🆕 Resultados de evaluación
└── best_meeple_cnn.pth              # 🆕 Modelo CNN entrenado
```

## 🎯 Flujo de Trabajo Recomendado

### ⭐ Paso 1: Sistema Integrado

```bash
# Iniciar el sistema completo
python meeple_system.py
```

Esto te da acceso a todas las herramientas:
1. **Anotación Manual** - Crear ground truth
2. **Evaluación** - Medir rendimiento del detector
3. **CNN** - Entrenar modelo de deep learning
4. **Configuración** - Ajustar parámetros
5. **Resultados** - Ver métricas y visualizaciones

### ⭐ Paso 2: Anotación Manual (Ground Truth)

Si el detector no funciona bien, crea datos ground truth:

```bash
# Opción 1 del sistema integrado, o:
python meeple_annotator.py
```

**Cómo usar:**
- Click izquierdo: marca posición de meeple (alternará azul/negro)
- ESPACIO: guardar y siguiente imagen
- Q: salir

Esto crea `manual_annotations.json` con las posiciones reales de los meeples.

### ⭐ Paso 3: Evaluar Rendimiento

```bash
# Opción 2 del sistema integrado, o:
python evaluate_detector.py
```

Compara el detector OpenCV vs tus anotaciones manuales:
- **Precisión**: ¿Qué porcentaje de detecciones son correctas?
- **Recall**: ¿Qué porcentaje de meeples reales detectó?
- **F1-Score**: Métrica balanceada
- **Visualizaciones**: Comparaciones lado a lado

### ⭐ Paso 4: Mejorar con CNN (Opcional)

Si quieres usar deep learning:

```bash
# Opción 3 del sistema integrado, o:
python cnn_meeple_detector.py
```

**Entrenamiento:**
- Usa tus anotaciones manuales como datos de entrenamiento
- Entrena una CNN para clasificar patches de meeples
- Evalúa rendimiento en datos no vistos

## 🔧 Detector OpenCV Mejorado

### Rangos HSV Actualizados

Basados en tus valores exactos:
```python
# Azul: HSV(212, 64%, 62%) ≈ H:106, S:163, V:158
'blue': {
    'lower': np.array([95, 140, 120]),
    'upper': np.array([115, 180, 190])
}

# Negro: HSV(240, 10%, 8%) ≈ H:120, S:26, V:20
'black': {
    'lower': np.array([0, 0, 0]),
    'upper': np.array([179, 50, 50])
}
```

### Cómo Probar Rápido

```bash
# Probar detector actualizado
python test_real_images.py
```

## 📊 Métricas de Evaluación

Después de crear anotaciones manuales, obtendrás:

```
📈 RESULTADOS GLOBALES:
Imágenes evaluadas: 12
Meeples ground truth: 12
Meeples detectados: 10
Correctos (posición): 8
False positives: 2
False negatives: 4
Precisión: 0.800
Recall: 0.667
F1-Score: 0.727
```

## 🎮 Ejemplos de Uso

### Anotación Interactiva
```bash
python meeple_annotator.py
# Click en imágenes -> crea manual_annotations.json
```

### Evaluación Completa
```bash
python evaluate_detector.py
# Compara detector vs ground truth -> evaluation_results.json
```

### Entrenamiento CNN
```bash
python cnn_meeple_detector.py
# Entrena modelo -> best_meeple_cnn.pth
```

## 🔍 Visualizaciones

El sistema genera:
- **Comparaciones lado a lado**: Ground truth vs Detección automática
- **Heatmaps de error**: Dónde falla el detector
- **Curvas de aprendizaje**: Para modelos CNN
- **Matrices de confusión**: Análisis detallado

## 🚀 Próximos Pasos

1. **Ejecuta** `python meeple_system.py`
2. **Crea anotaciones** para algunas imágenes problemáticas
3. **Evalúa** el rendimiento actual
4. **Decide** si usar CNN o mejorar OpenCV
5. **Itera** hasta obtener precisión satisfactoria

## 💡 Consejos

- **Empieza pequeño**: Anota 5-10 imágenes primero
- **CNN vs OpenCV**: CNN es más precisa pero requiere más datos
- **Iluminación**: El detector funciona mejor con iluminación consistente
- **Calidad**: Usa fotos nítidas con la loseta bien centrada

¡El sistema está diseñado para mejorar iterativamente con tus datos reales!

## 🚀 Características

- ✅ **Detección automática** de meeples circulares (1 por imagen)
- ✅ **Clasificación de colores** precisa (azul/negro únicamente)
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