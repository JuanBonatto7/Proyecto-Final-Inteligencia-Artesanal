# 🎮 Sistema de Detección de Tablero de Carcassonne

Sistema completo de visión por computadora para detectar, identificar y reconstruir el estado de un tablero del juego Carcassonne a partir de una fotografía.

## 📋 Características

- ✅ Detección automática de fichas en el tablero
- ✅ Clasificación de tipo de ficha (A-Y)
- ✅ Detección de rotación (0, 1, 2, 3)
- ✅ Corrección de perspectiva automática
- ✅ Exportación de matriz del tablero
- ✅ Visualización paso a paso del procesamiento
- ✅ Arquitectura modular y extensible

## 🏗️ Arquitectura

El sistema utiliza un **Pipeline** de procesamiento secuencial con 10 pasos:

1. **Resize**: Redimensiona imagen para optimizar procesamiento
2. **Blur**: Reduce ruido con filtro gaussiano
3. **Canny Edge Detection**: Detecta bordes
4. **Dilate**: Fortalece bordes detectados
5. **Hough Line Transform**: Detecta líneas rectas
6. **Find Intersections**: Encuentra puntos de intersección (esquinas)
7. **RANSAC Homography**: Corrige perspectiva
8. **Find Tiles**: Localiza y extrae cada ficha
9. **Tile Classifier**: Clasifica tipo y rotación
10. **Reconstruct Board**: Reconstruye matriz del tablero

## 📦 Instalación

### Requisitos

- Python 3.8+
- pip

### Pasos

```bash
# Clonar el repositorio
git clone <repo-url>
cd carcassonne_detector

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt