"""
DETECTOR DE LOSETAS CARCASSONNE v9.0
Estrategia completamente nueva: Detección por patrones y segmentación de color

pip install opencv-python numpy matplotlib scikit-image scipy scikit-learn

Uso: python carcassonne_detector.py imagen.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal, ndimage
from scipy.fft import fft2, fftshift
from sklearn.cluster import DBSCAN
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import warnings
warnings.filterwarnings('ignore')

class DetectorCarcassonne:
    def __init__(self, imagen_path):
        self.imagen_path = imagen_path
        self.imagen_original = None
        self.losetas = []
        self.debug_mode = True
        
    def cargar_imagen(self):
        """Carga la imagen"""
        self.imagen_original = cv2.imread(self.imagen_path)
        if self.imagen_original is None:
            raise ValueError(f"No se pudo cargar: {self.imagen_path}")
        print(f"✓ Imagen cargada: {self.imagen_original.shape}")
        return self.imagen_original
    
    def estimar_tamano_loseta(self):
        """
        Estima el tamaño de las losetas usando autocorrelación
        para encontrar patrones repetitivos
        """
        print("\n📐 Estimando tamaño de losetas...")
        
        gray = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Método 1: Usando transformada de Fourier para encontrar periodicidad
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Buscar picos en el espectro (indicadores de periodicidad)
        # Pero esto puede ser complejo, así que usemos un método más simple
        
        # Método 2: Detección de bordes y análisis de distancias
        edges = cv2.Canny(gray, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área para obtener posibles losetas
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000 and area < (h * w) / 20:  # Área razonable para una loseta
                areas.append(area)
        
        if areas:
            area_mediana = np.median(areas)
            lado_estimado = np.sqrt(area_mediana)
        else:
            # Estimación por defecto: dividir el área total por 72 losetas
            lado_estimado = np.sqrt((h * w) / 72)
        
        # Método 3: Análisis de la imagen para detectar regiones de color uniforme
        # Convertir a LAB para mejor separación de colores
        lab = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2LAB)
        
        # Aplicar K-means para segmentar colores principales
        Z = lab.reshape((-1, 3))
        from sklearn.cluster import KMeans
        
        # Usar menos clusters para velocidad
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        kmeans.fit(Z)
        labels = kmeans.labels_.reshape((h, w))
        
        # Analizar componentes conectados
        tamaños_componentes = []
        for i in range(8):
            mask = (labels == i).astype(np.uint8) * 255
            num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            
            for j in range(1, num_labels):
                area = stats[j, cv2.CC_STAT_AREA]
                if 5000 < area < (h * w) / 20:
                    tamaños_componentes.append(area)
        
        if tamaños_componentes:
            area_tipica = np.median(tamaños_componentes)
            lado_estimado_2 = np.sqrt(area_tipica)
            lado_estimado = (lado_estimado + lado_estimado_2) / 2
        
        print(f"  ✓ Tamaño estimado: {lado_estimado:.1f}x{lado_estimado:.1f} píxeles")
        
        return int(lado_estimado)
    
    def segmentar_por_color(self):
        """
        Segmenta la imagen por color para separar losetas del fondo
        """
        print("\n🎨 Segmentando por color...")
        
        # Convertir a HSV para mejor segmentación
        hsv = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2HSV)
        
        # El fondo suele ser gris, las losetas tienen colores más saturados
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Crear máscara: las losetas tienen mayor saturación que el fondo gris
        mask_losetas = (saturation > 30) | (value < 100) | (value > 200)
        mask_losetas = mask_losetas.astype(np.uint8) * 255
        
        # Limpiar la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_losetas = cv2.morphologyEx(mask_losetas, cv2.MORPH_CLOSE, kernel)
        mask_losetas = cv2.morphologyEx(mask_losetas, cv2.MORPH_OPEN, kernel)
        
        if self.debug_mode:
            cv2.imwrite('debug_01_mask_color.png', mask_losetas)
        
        print("  ✓ Máscara de color creada")
        
        return mask_losetas
    
    def detectar_usando_watershed(self, lado_estimado):
        """
        Usa watershed para segmentar las losetas
        """
        print("\n💧 Aplicando segmentación Watershed...")
        
        gray = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        
        # Aplicar threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calcular distancia transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Encontrar picos locales (centros de losetas)
        # Obtener coordenadas de picos locales (peak_local_max devuelve coordenadas en versiones recientes)
        coords = peak_local_max(
            dist_transform,
            min_distance=int(lado_estimado * 0.5),
            threshold_abs=0.0,
            exclude_border=False
        )
        
        # Crear máscara booleana de picos y luego marcadores etiquetados
        local_maxima = np.zeros_like(dist_transform, dtype=bool)
        if coords is not None and len(coords) > 0:
            # coords están en formato (row, col)
            local_maxima[tuple(coords.T)] = True

        markers = ndimage.label(local_maxima)[0]
        
        # Aplicar watershed
        labels = watershed(-dist_transform, markers, mask=binary)
        
        # Extraer regiones
        losetas_watershed = []
        for label_id in range(1, np.max(labels) + 1):
            mask = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = contours[0]
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filtrar por tamaño
                area_min = (lado_estimado * 0.5) ** 2
                area_max = (lado_estimado * 1.5) ** 2
                
                if area_min < area < area_max:
                    losetas_watershed.append({
                        'bbox': (x, y, w, h),
                        'centro': (x + w // 2, y + h // 2),
                        'area': area,
                        'metodo': 'watershed'
                    })
        
        print(f"  ✓ {len(losetas_watershed)} regiones encontradas")
        
        return losetas_watershed
    
    def detectar_usando_ventana_deslizante(self, lado_estimado):
        """
        Usa ventana deslizante para buscar losetas
        """
        print("\n🔍 Aplicando ventana deslizante...")
        
        gray = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Parámetros de la ventana
        window_size = int(lado_estimado * 0.9)
        step_size = int(lado_estimado * 0.3)  # Superposición para no perder losetas
        
        candidatos = []
        
        for y in range(0, h - window_size, step_size):
            for x in range(0, w - window_size, step_size):
                # Extraer ventana
                window = gray[y:y+window_size, x:x+window_size]
                
                # Calcular características
                mean_val = np.mean(window)
                std_val = np.std(window)
                
                # Calcular textura usando gradientes
                dx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
                dy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(dx**2 + dy**2)
                texture_score = np.mean(gradient_magnitude)
                
                # Una loseta debe tener:
                # - Variación de intensidad (std > 20)
                # - No ser muy brillante (mean < 140)
                # - Textura significativa
                if std_val > 20 and mean_val < 140 and texture_score > 10:
                    candidatos.append({
                        'bbox': (x, y, window_size, window_size),
                        'centro': (x + window_size // 2, y + window_size // 2),
                        'area': window_size ** 2,
                        'score': std_val * texture_score,
                        'metodo': 'sliding_window'
                    })
        
        # Eliminar superposiciones usando Non-Maximum Suppression
        candidatos = self.nms(candidatos, threshold=0.3)
        
        print(f"  ✓ {len(candidatos)} candidatos encontrados")
        
        return candidatos
    
    def detectar_usando_contornos(self, lado_estimado):
        """
        Detecta losetas buscando contornos cuadrados
        """
        print("\n📦 Detectando contornos cuadrados...")
        
        gray = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        
        # Aplicar múltiples métodos de detección de bordes
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 100, 200)
        
        # Combinar todos los bordes
        edges_combined = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Dilatar para conectar bordes cercanos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges_combined, kernel, iterations=1)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        losetas_contornos = []
        
        for contour in contours:
            # Aproximar contorno a polígono
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Obtener bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Calcular qué tan cuadrado es
            if w > 0 and h > 0:
                aspect_ratio = float(w) / h
                extent = area / (w * h)
                
                # Filtros para losetas válidas
                area_min = (lado_estimado * 0.4) ** 2
                area_max = (lado_estimado * 1.6) ** 2
                
                # Una loseta debe ser aproximadamente cuadrada
                if (0.7 < aspect_ratio < 1.3 and 
                    area_min < area < area_max and
                    extent > 0.5):  # El contorno llena al menos 50% del bounding box
                    
                    # Verificar contenido
                    region = gray[y:y+h, x:x+w]
                    if region.size > 0:
                        mean_val = np.mean(region)
                        std_val = np.std(region)
                        
                        if std_val > 15 and mean_val < 150:
                            losetas_contornos.append({
                                'bbox': (x, y, w, h),
                                'centro': (x + w // 2, y + h // 2),
                                'area': area,
                                'metodo': 'contornos'
                            })
        
        print(f"  ✓ {len(losetas_contornos)} contornos válidos")
        
        return losetas_contornos
    
    def detectar_usando_template_matching(self, lado_estimado):
        """
        Busca patrones similares a losetas usando template matching
        """
        print("\n🎯 Aplicando Template Matching...")
        
        gray = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Primero, intentar encontrar una loseta de ejemplo
        # Buscar en el centro de la imagen donde es más probable que haya losetas
        cx, cy = w // 2, h // 2
        template_size = int(lado_estimado * 0.8)
        
        # Extraer múltiples templates de diferentes partes
        templates = []
        offsets = [
            (0, 0), 
            (-lado_estimado, 0), (lado_estimado, 0),
            (0, -lado_estimado), (0, lado_estimado)
        ]
        
        for dx, dy in offsets:
            x = cx + dx - template_size // 2
            y = cy + dy - template_size // 2
            
            if 0 <= x < w - template_size and 0 <= y < h - template_size:
                template = gray[y:y+template_size, x:x+template_size]
                
                # Verificar que el template tiene contenido
                if np.std(template) > 20:
                    templates.append(template)
        
        if not templates:
            print("  ⚠️ No se encontraron templates válidos")
            return []
        
        # Aplicar template matching con cada template
        all_matches = []
        
        for i, template in enumerate(templates):
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.6
            loc = np.where(result >= threshold)
            
            for pt in zip(*loc[::-1]):
                all_matches.append({
                    'bbox': (pt[0], pt[1], template_size, template_size),
                    'centro': (pt[0] + template_size // 2, pt[1] + template_size // 2),
                    'area': template_size ** 2,
                    'score': result[pt[1], pt[0]],
                    'metodo': 'template'
                })
        
        # Eliminar duplicados
        all_matches = self.nms(all_matches, threshold=0.5)
        
        print(f"  ✓ {len(all_matches)} matches encontrados")
        
        return all_matches
    
    def detectar_usando_clustering(self, lado_estimado):
        """
        Usa clustering DBSCAN para agrupar píxeles en losetas
        """
        print("\n🌟 Aplicando clustering DBSCAN...")
        
        # Reducir resolución para velocidad
        scale = 0.25
        small = cv2.resize(self.imagen_original, None, fx=scale, fy=scale)
        
        # Convertir a LAB
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]
        
        # Crear características: posición + color
        features = []
        positions = []
        
        for y in range(h):
            for x in range(w):
                # Solo considerar píxeles no grises
                l, a, b = lab[y, x]
                saturation = np.sqrt(a**2 + b**2)
                
                if saturation > 10 or l < 100:  # No es gris neutro
                    # Normalizar posición y color
                    features.append([
                        x / w * 10,  # Peso a la posición
                        y / h * 10,
                        l / 255,
                        a / 255,
                        b / 255
                    ])
                    positions.append((x, y))
        
        if len(features) < 100:
            print("  ⚠️ Muy pocos píxeles no grises")
            return []
        
        features = np.array(features)
        
        # Aplicar DBSCAN
        eps = 0.5
        min_samples = 50
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        
        # Extraer clusters
        losetas_clusters = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remover ruido
        
        for label in unique_labels:
            cluster_positions = [positions[i] for i in range(len(labels)) if labels[i] == label]
            
            if cluster_positions:
                xs = [p[0] for p in cluster_positions]
                ys = [p[1] for p in cluster_positions]
                
                # Convertir a coordenadas originales
                x_min = int(min(xs) / scale)
                x_max = int(max(xs) / scale)
                y_min = int(min(ys) / scale)
                y_max = int(max(ys) / scale)
                
                w = x_max - x_min
                h = y_max - y_min
                area = w * h
                
                # Filtrar por tamaño
                area_min = (lado_estimado * 0.4) ** 2
                area_max = (lado_estimado * 1.6) ** 2
                
                if area_min < area < area_max:
                    losetas_clusters.append({
                        'bbox': (x_min, y_min, w, h),
                        'centro': ((x_min + x_max) // 2, (y_min + y_max) // 2),
                        'area': area,
                        'metodo': 'clustering'
                    })
        
        print(f"  ✓ {len(losetas_clusters)} clusters válidos")
        
        return losetas_clusters
    
    def nms(self, detecciones, threshold=0.5):
        """
        Non-Maximum Suppression para eliminar detecciones duplicadas
        """
        if not detecciones:
            return []
        
        # Convertir a arrays
        boxes = np.array([d['bbox'] for d in detecciones])
        scores = np.array([d.get('score', 1.0) for d in detecciones])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return [detecciones[i] for i in keep]
    
    def fusionar_detecciones(self, todas_detecciones, lado_estimado):
        """
        Fusiona todas las detecciones de diferentes métodos
        """
        print("\n🔀 Fusionando todas las detecciones...")
        
        # Combinar todas
        todas = []
        for detecciones in todas_detecciones:
            todas.extend(detecciones)
        
        print(f"  → Total antes de fusión: {len(todas)}")
        
        # Aplicar NMS global
        fusionadas = self.nms(todas, threshold=0.4)
        
        print(f"  → Después de NMS: {len(fusionadas)}")
        
        # Refinar: verificar que cada detección es válida
        validas = []
        gray = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        
        for det in fusionadas:
            x, y, w, h = det['bbox']
            
            # Verificar límites
            if x >= 0 and y >= 0 and x + w <= gray.shape[1] and y + h <= gray.shape[0]:
                region = gray[y:y+h, x:x+w]
                
                if region.size > 0:
                    mean_val = np.mean(region)
                    std_val = np.std(region)
                    
                    # Verificar que tiene contenido válido
                    if std_val > 10 and mean_val < 160:
                        det['id'] = len(validas)
                        validas.append(det)
        
        print(f"  ✓ {len(validas)} detecciones válidas finales")
        
        return validas
    
    def procesar(self):
        """Proceso completo con múltiples métodos"""
        print("╔════════════════════════════════════════════╗")
        print("║   DETECTOR CARCASSONNE v9.0               ║")
        print("║   Multi-método: Sin depender de líneas     ║")
        print("╚════════════════════════════════════════════╝\n")
        
        # 1. Cargar
        print("📂 Paso 1: Cargando imagen...")
        self.cargar_imagen()
        
        # 2. Estimar tamaño
        print("\n📐 Paso 2: Estimando dimensiones...")
        lado_estimado = self.estimar_tamano_loseta()
        
        # 3. Aplicar múltiples métodos de detección
        todas_detecciones = []
        
        print("\n🔬 Paso 3: Aplicando múltiples métodos de detección...")
        
        # Método 1: Contornos
        detecciones_contornos = self.detectar_usando_contornos(lado_estimado)
        todas_detecciones.append(detecciones_contornos)
        
        # Método 2: Ventana deslizante
        detecciones_ventana = self.detectar_usando_ventana_deslizante(lado_estimado)
        todas_detecciones.append(detecciones_ventana)
        
        # Método 3: Template matching
        detecciones_template = self.detectar_usando_template_matching(lado_estimado)
        todas_detecciones.append(detecciones_template)
        
        # Método 4: Watershed
        detecciones_watershed = self.detectar_usando_watershed(lado_estimado)
        todas_detecciones.append(detecciones_watershed)
        
        # Método 5: Clustering
        detecciones_clustering = self.detectar_usando_clustering(lado_estimado)
        todas_detecciones.append(detecciones_clustering)
        
        # 4. Fusionar todas las detecciones
        print("\n🔀 Paso 4: Fusionando resultados...")
        self.losetas = self.fusionar_detecciones(todas_detecciones, lado_estimado)
        
        # 5. Completar huecos si es necesario
        if len(self.losetas) < 72:
            print("\n🔧 Paso 5: Buscando losetas faltantes...")
            self.completar_faltantes(lado_estimado)
        
        # 6. Visualizar
        print("\n🎨 Paso 6: Generando visualización...")
        self.visualizar()
        
        # 7. Extraer
        print("\n💾 Paso 7: Extrayendo losetas individuales...")
        self.extraer_imagenes()
        
        # 8. Reporte
        self.generar_reporte()
        
        return self.losetas
    
    def completar_faltantes(self, lado_estimado):
        """
        Intenta encontrar losetas faltantes en áreas no cubiertas
        """
        h, w = self.imagen_original.shape[:2]
        
        # Crear máscara de áreas ya cubiertas
        mask_cubierta = np.zeros((h, w), dtype=np.uint8)
        
        for loseta in self.losetas:
            x, y, bw, bh = loseta['bbox']
            mask_cubierta[y:y+bh, x:x+bw] = 255
        
        # Buscar áreas no cubiertas
        mask_no_cubierta = 255 - mask_cubierta
        
        # Buscar componentes conectados en áreas no cubiertas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_no_cubierta, connectivity=8
        )
        
        gray = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2GRAY)
        nuevas_losetas = []
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Si el área es suficientemente grande
            if area > (lado_estimado * 0.5) ** 2:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Verificar contenido
                region = gray[y:y+h, x:x+w]
                if region.size > 0:
                    mean_val = np.mean(region)
                    std_val = np.std(region)
                    
                    if std_val > 10 and mean_val < 160:
                        nuevas_losetas.append({
                            'bbox': (x, y, w, h),
                            'centro': (x + w // 2, y + h // 2),
                            'area': area,
                            'metodo': 'completado',
                            'id': len(self.losetas) + len(nuevas_losetas)
                        })
        
        if nuevas_losetas:
            print(f"  ✓ {len(nuevas_losetas)} losetas adicionales encontradas")
            self.losetas.extend(nuevas_losetas)
    
    def visualizar(self):
        """Visualiza las detecciones"""
        img_vis = self.imagen_original.copy()
        
        # Colores por método
        colores_metodo = {
            'contornos': (255, 0, 0),      # Azul
            'sliding_window': (0, 255, 0),  # Verde
            'template': (0, 165, 255),      # Naranja
            'watershed': (255, 0, 255),     # Magenta
            'clustering': (255, 255, 0),    # Cyan
            'completado': (128, 0, 255)     # Violeta
        }
        
        for loseta in self.losetas:
            x, y, w, h = loseta['bbox']
            metodo = loseta.get('metodo', 'unknown')
            color = colores_metodo.get(metodo, (255, 255, 255))
            
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), color, 2)
            
            # Número de ID
            cv2.putText(
                img_vis,
                str(loseta['id']),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        cv2.imwrite('deteccion_final.png', img_vis)
        
        # Crear visualización con matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Imagen original
        ax1.imshow(cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB))
        ax1.set_title('Imagen Original', fontsize=16)
        ax1.axis('off')
        
        # Imagen con detecciones
        ax2.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Detecciones: {len(self.losetas)} losetas', fontsize=16)
        ax2.axis('off')
        
        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Contornos'),
            Patch(facecolor='green', label='Ventana Deslizante'),
            Patch(facecolor='orange', label='Template'),
            Patch(facecolor='magenta', label='Watershed'),
            Patch(facecolor='yellow', label='Clustering'),
            Patch(facecolor='violet', label='Completado')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('visualizacion_completa.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def extraer_imagenes(self, output_dir='losetas_extraidas'):
        """Extrae las losetas individuales"""
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        for loseta in self.losetas:
            x, y, w, h = loseta['bbox']
            
            # Asegurar límites
            x = max(0, x)
            y = max(0, y)
            h_img, w_img = self.imagen_original.shape[:2]
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            if w > 0 and h > 0:
                loseta_img = self.imagen_original[y:y+h, x:x+w]
                metodo = loseta.get('metodo', 'unknown')
                filename = f"loseta_{loseta['id']:03d}_{metodo}.png"
                cv2.imwrite(os.path.join(output_dir, filename), loseta_img)
    
    def generar_reporte(self):
        """Genera reporte final"""
        print("\n" + "="*50)
        print("                 REPORTE FINAL")
        print("="*50)
        
        print(f"\n📊 Resultados:")
        print(f"  • Total detectado: {len(self.losetas)} / 72")
        print(f"  • Precisión: {len(self.losetas)/72*100:.1f}%")
        
        # Contar por método
        from collections import Counter
        metodos = Counter([l.get('metodo', 'unknown') for l in self.losetas])
        
        print(f"\n📈 Por método:")
        for metodo, count in metodos.most_common():
            print(f"  • {metodo}: {count}")
        
        print(f"\n📁 Archivos generados:")
        print(f"  • deteccion_final.png")
        print(f"  • visualizacion_completa.png")
        print(f"  • losetas_extraidas/")
        
        print("\n" + "="*50)
        print("✅ PROCESO COMPLETADO")
        print("="*50)


def main():
    if len(sys.argv) < 2:
        print("Uso: python detector.py <imagen>")
        return
    
    detector = DetectorCarcassonne(sys.argv[1])
    
    try:
        detector.procesar()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()