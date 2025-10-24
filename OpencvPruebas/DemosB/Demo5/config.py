"""
Configuración global del sistema de detección de Carcassonne.
Basado en el proyecto original de Carcassonne tracking.
"""

from typing import Tuple


class Config:
    """Configuración centralizada del sistema."""
    
    # ============= TILE EXTRACTION =============
    TILE_SIZE: int = 64  # ✅
    GRID_RANGE_MIN: int = -10  # ✅
    GRID_RANGE_MAX: int = 10  # ✅
    MIN_EDGE_DENSITY: float = 8.0  # ✅

    # ============= CONFIGURACIÓN DE IMAGEN =============
    MAX_IMAGE_WIDTH: int = 800
    
    # ============= PREPROCESSING =============
    # Blur (Paso 1)
    BLUR_KERNEL_SIZE: Tuple[int, int] = (9, 9)  # Kernel más grande como en original
    BLUR_SIGMA: float = 0.75
    
    # ============= EDGE DETECTION =============
    # Canny (Paso 2)
    CANNY_THRESHOLD_1: int = 50
    CANNY_THRESHOLD_2: int = 150
    CANNY_APERTURE_SIZE: int = 3
    
    # Dilate (Paso 3)
    DILATE_KERNEL_SIZE: Tuple[int, int] = (2, 2)  # Kernel pequeño como en original
    DILATE_ITERATIONS: int = 1
    
    # ============= LINE DETECTION =============
    # Hough Line Transform Probabilístico (Paso 4)
    HOUGH_RHO: float = 1.0
    HOUGH_THETA: float = 3.14159265 / 180
    HOUGH_THRESHOLD: int = 80  # Umbral para líneas fuertes
    HOUGH_MIN_LINE_LENGTH: int = 50  # Líneas mínimas más largas
    HOUGH_MAX_LINE_GAP: int = 20  # Gap permitido entre segmentos
    
    # ============= FIND INTERSECTIONS (Paso 5) =============
    # Clasificación Vertical/Horizontal (KMeans)
    KMEANS_N_CLUSTERS: int = 2  # Siempre 2: vertical y horizontal
    KMEANS_RANDOM_STATE: int = 42
    KMEANS_N_INIT: int = 10
    
    # Filtrado de Outliers (RANSAC simple)
    OUTLIER_RANSAC_ITERATIONS: int = 50
    OUTLIER_ANGLE_THRESHOLD_DEG: float = 10.0  # ±10 grados de tolerancia
    
    # Agrupamiento de Líneas Paralelas (MeanShift)
    MEANSHIFT_BANDWIDTH: float = 20.0  # Ancho de banda para agrupar líneas
    
    # Votación de Intersecciones
    VOTE_MIN_DISTANCE: float = 10.0  # Distancia mínima para evitar división por 0
    
    # Non-Maximum Suppression
    NMS_WINDOW_SIZE: int = 25  # Ventana para eliminar duplicados
    
    # ============= RANSAC HOMOGRAPHY (Paso 6) =============
    # Tamaño de loseta (tile) en píxeles
    TILE_SIZE: int = 64  # Estándar del proyecto original
    
    # RANSAC para homografía
    RANSAC_THRESHOLD: float = 5.0  # Threshold en píxeles
    RANSAC_MAX_ITERATIONS: int = 1000
    RANSAC_CONFIDENCE: float = 0.99
    
    # Mínimo de puntos para calcular homografía
    MIN_HOMOGRAPHY_POINTS: int = 4
    
    # ============= TILE EXTRACTION =============
    # Rango de grilla para extraer tiles
    GRID_RANGE_MIN: int = -10  # Buscar desde (-10, -10)
    GRID_RANGE_MAX: int = 10   # Hasta (10, 10)
    
    # Validación de tiles
    MIN_EDGE_DENSITY: float = 8.0  # Densidad mínima de bordes para considerar tile válido
    
    # ============= TILE CLASSIFICATION =============
    # Template Matching
    TEMPLATE_MATCHING_METHOD: int = 5  # cv2.TM_CCOEFF_NORMED
    MIN_MATCH_SCORE: float = 0.45  # Umbral mínimo de confianza
    
    # Feature Matching (alternativo)
    FEATURE_DESCRIPTOR: str = "ORB"  # ORB, SIFT, AKAZE
    ORB_N_FEATURES: int = 500
    ORB_FAST_THRESHOLD: int = 5
    MATCH_RATIO_THRESHOLD: float = 0.7  # Ratio test de Lowe
    MIN_MATCHES_THRESHOLD: int = 8
    
    # ============= VISUALIZACIÓN =============
    # Colores para debug (BGR)
    COLOR_INLIERS: Tuple[int, int, int] = (0, 255, 0)  # Verde
    COLOR_OUTLIERS: Tuple[int, int, int] = (0, 0, 255)  # Rojo
    COLOR_GRID: Tuple[int, int, int] = (255, 255, 0)  # Amarillo
    COLOR_INTERSECTIONS: Tuple[int, int, int] = (0, 255, 255)  # Cyan
    
    # Paleta de colores para grupos de líneas
    LINE_COLORS = [
        (255, 0, 0),    # Azul
        (0, 255, 0),    # Verde
        (0, 0, 255),    # Rojo
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarillo
        (128, 0, 128),  # Púrpura
        (255, 128, 0),  # Naranja
        (0, 128, 255),  # Azul claro
        (128, 255, 0),  # Verde lima
    ]
    
    # Ventanas de visualización
    VISUALIZATION_WINDOW_NAME: str = "Carcassonne Detector"
    VISUALIZATION_WAIT_TIME: int = 0  # 0 = esperar tecla
    VISUALIZATION_MAX_HEIGHT: int = 600
    
    # ============= RUTAS =============
    TILES_TEMPLATE_PATH: str = "locetas_referencia"
    OUTPUT_FOLDER: str = "output"
    DEBUG_FOLDER: str = "output/debug"
    
    # ============= DEBUG =============
    DEBUG_MODE: bool = True
    SHOW_INTERMEDIATE_STEPS: bool = True
    SAVE_DEBUG_IMAGES: bool = True
    VERBOSE: bool = True
    
    # Nombres de imágenes de debug
    DEBUG_NAMES = {
        'blur': '01_blur.jpg',
        'edges': '02_canny.jpg',
        'dilated': '03_dilated.jpg',
        'lines': '04_hough_lines.jpg',
        'groups': '05_line_groups.jpg',
        'intersections': '06_intersections.jpg',
        'homography': '07_homography.jpg',
        'tiles': '08_extracted_tiles/',
        'classification': '09_classification.jpg',
        'final': 'resultado_final.jpg'
    }


# ============= CONSTANTES DEL JUEGO =============
# Letras de losetas del Carcassonne (A-Y, 24 tipos)
TILE_TYPES = [chr(i) for i in range(ord('A'), ord('Y') + 1)]

# Rotaciones posibles (0, 1, 2, 3 = 0°, 90°, 180°, 270°)
ROTATIONS = [0, 1, 2, 3]

# Mapeo de rotación a grados
ROTATION_TO_DEGREES = {
    0: 0,
    1: 90,
    2: 180,
    3: 270
}


# ============= FUNCIONES HELPER =============
def get_debug_path(key: str) -> str:
    """Retorna path completo de imagen de debug"""
    from pathlib import Path
    debug_folder = Path(Config.DEBUG_FOLDER)
    debug_folder.mkdir(parents=True, exist_ok=True)
    return str(debug_folder / Config.DEBUG_NAMES.get(key, f'{key}.jpg'))


def print_config_summary():
    """Imprime resumen de configuración"""
    print("\n" + "="*70)
    print("⚙️  CONFIGURACIÓN DEL SISTEMA")
    print("="*70)
    print(f"\n📐 Procesamiento de Imagen:")
    print(f"   Max Width: {Config.MAX_IMAGE_WIDTH}px")
    print(f"   Blur Kernel: {Config.BLUR_KERNEL_SIZE}")
    
    print(f"\n🔍 Detección de Líneas:")
    print(f"   Hough Threshold: {Config.HOUGH_THRESHOLD}")
    print(f"   Min Line Length: {Config.HOUGH_MIN_LINE_LENGTH}px")
    print(f"   Max Line Gap: {Config.HOUGH_MAX_LINE_GAP}px")
    
    print(f"\n📍 Intersecciones:")
    print(f"   Outlier Threshold: {Config.OUTLIER_ANGLE_THRESHOLD_DEG}°")
    print(f"   MeanShift Bandwidth: {Config.MEANSHIFT_BANDWIDTH}")
    print(f"   NMS Window: {Config.NMS_WINDOW_SIZE}px")
    
    print(f"\n🎯 Homografía:")
    print(f"   Tile Size: {Config.TILE_SIZE}px")
    print(f"   RANSAC Threshold: {Config.RANSAC_THRESHOLD}px")
    print(f"   Min Points: {Config.MIN_HOMOGRAPHY_POINTS}")
    
    print(f"\n🧩 Clasificación:")
    print(f"   Min Match Score: {Config.MIN_MATCH_SCORE}")
    print(f"   Min Edge Density: {Config.MIN_EDGE_DENSITY}")
    
    print(f"\n📁 Rutas:")
    print(f"   Templates: {Config.TILES_TEMPLATE_PATH}")
    print(f"   Debug: {Config.DEBUG_FOLDER}")
    
    print(f"\n🐛 Debug:")
    print(f"   Mode: {'ON' if Config.DEBUG_MODE else 'OFF'}")
    print(f"   Save Images: {'YES' if Config.SAVE_DEBUG_IMAGES else 'NO'}")
    print("="*70 + "\n")


# ============= VALIDACIÓN =============
def validate_config():
    """Valida que la configuración sea coherente"""
    errors = []
    
    # Validar tamaños
    if Config.MAX_IMAGE_WIDTH < 400:
        errors.append("MAX_IMAGE_WIDTH debe ser >= 400")
    
    if Config.TILE_SIZE < 32 or Config.TILE_SIZE > 256:
        errors.append("TILE_SIZE debe estar entre 32 y 256")
    
    # Validar kernels
    if Config.BLUR_KERNEL_SIZE[0] % 2 == 0 or Config.BLUR_KERNEL_SIZE[1] % 2 == 0:
        errors.append("BLUR_KERNEL_SIZE debe tener valores impares")
    
    # Validar thresholds
    if Config.RANSAC_THRESHOLD <= 0:
        errors.append("RANSAC_THRESHOLD debe ser > 0")
    
    if Config.MIN_MATCH_SCORE < 0 or Config.MIN_MATCH_SCORE > 1:
        errors.append("MIN_MATCH_SCORE debe estar entre 0 y 1")
    
    # Validar rutas
    from pathlib import Path
    if not Path(Config.TILES_TEMPLATE_PATH).exists():
        errors.append(f"Carpeta de templates no existe: {Config.TILES_TEMPLATE_PATH}")
    
    if errors:
        print("\n⚠️  ERRORES EN CONFIGURACIÓN:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    return True


# ============= AUTO-EJECUCIÓN =============
if __name__ == "__main__":
    print_config_summary()
    
    if validate_config():
        print("✅ Configuración válida\n")
    else:
        print("\n❌ Configuración con errores\n")