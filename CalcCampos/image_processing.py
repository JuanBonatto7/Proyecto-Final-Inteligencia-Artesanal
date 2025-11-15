import cv2
import numpy as np

TOLERANCE = 10

def segment_colors(img, colors):
    """Segmenta la imagen por colores con tolerancia"""
    masks = {}
    
    for name, color in colors.items():
        # Convertir BGR a RGB si es necesario
        lower = np.array([max(0, c - TOLERANCE) for c in color[::-1]])
        upper = np.array([min(255, c + TOLERANCE) for c in color[::-1]])
        
        mask = cv2.inRange(img, lower, upper)
        masks[name] = mask
    
    return masks

def find_components(mask):
    """Encuentra componentes conectados en una máscara"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    
    components = []
    for i in range(1, num_labels):  # Saltar fondo (0)
        component = {
            'id': i,
            'mask': (labels == i).astype(np.uint8) * 255,
            'area': stats[i, cv2.CC_STAT_AREA],
            'centroid': centroids[i],
            'bbox': (stats[i, cv2.CC_STAT_LEFT], 
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH], 
                    stats[i, cv2.CC_STAT_HEIGHT])
        }
        components.append(component)
    
    return components

def touches_mask(component_mask, target_mask):
    """Verifica si una componente toca otra máscara (4-conexión)"""
    # Dilatar ligeramente la componente para detectar 4-conexión
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    dilated = cv2.dilate(component_mask, kernel, iterations=1)
    
    # Ver si hay intersección
    intersection = cv2.bitwise_and(dilated, target_mask)
    return np.any(intersection > 0)

def touches_border(mask, img_shape):
    """Verifica si una máscara toca el borde de la imagen"""
    h, w = img_shape[:2]
    
    # Verificar bordes
    if np.any(mask[0, :] > 0):  # Borde superior
        return True
    if np.any(mask[h-1, :] > 0):  # Borde inferior
        return True
    if np.any(mask[:, 0] > 0):  # Borde izquierdo
        return True
    if np.any(mask[:, w-1] > 0):  # Borde derecho
        return True
    
    return False

def find_fields(field_mask):
    """Encuentra todos los campos (regiones verdes)"""
    fields = find_components(field_mask)
    
    # Agregar ID secuencial
    for i, field in enumerate(fields, 1):
        field['field_id'] = i
    
    return fields

def find_castles(castle_mask, white_mask, img_shape):
    """Encuentra castillos y determina cuáles están cerrados"""
    castles = find_components(castle_mask)
    
    for i, castle in enumerate(castles, 1):
        castle['castle_id'] = i
        
        # Un castillo está cerrado si toca blanco O borde
        touches_white = touches_mask(castle['mask'], white_mask)
        touches_edge = touches_border(castle['mask'], img_shape)
        
        castle['closed'] = touches_white or touches_edge
    
    return castles

def detect_meeples(masks, img_shape):
    """Detecta meeples de ambos jugadores y los filtra"""
    meeples_data = {1: [], 2: []}
    
    # Detectar meeples jugador 1 (violeta)
    meeples_1 = find_components(masks['MEEPLE_1'])
    for meeple in meeples_1:
        # Ignorar si toca camino (ROAD)
        if touches_mask(meeple['mask'], masks['ROAD']):
            continue
        
        meeple['player'] = 1
        meeples_data[1].append(meeple)
    
    # Detectar meeples jugador 2 (negro)
    meeples_2 = find_components(masks['MEEPLE_2'])
    for meeple in meeples_2:
        # Ignorar si toca camino (ROAD)
        if touches_mask(meeple['mask'], masks['ROAD']):
            continue
        
        meeple['player'] = 2
        meeples_data[2].append(meeple)
    
    return meeples_data

def create_visualization_masks(masks, img_shape):
    """Crea visualización de todas las máscaras"""
    h, w = img_shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Colores para visualización (BGR)
    vis[masks['FIELD'] > 0] = [76, 177, 34]      # Verde
    vis[masks['CASTLE'] > 0] = [39, 127, 255]    # Naranja
    vis[masks['ROAD'] > 0] = [204, 72, 63]       # Azul
    vis[masks['VERTEX'] > 0] = [36, 28, 237]     # Rojo
    vis[masks['MEEPLE_1'] > 0] = [164, 73, 163]  # Violeta
    vis[masks['MEEPLE_2'] > 0] = [0, 0, 0]       # Negro
    vis[masks['TILE_BG'] > 0] = [255, 255, 255]  # Blanco
    
    return vis

def create_fields_labeled(img, fields):
    """Crea visualización con campos etiquetados"""
    vis = img.copy()
    
    # Crear overlay transparente
    overlay = vis.copy()
    
    # Colores para cada campo
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
        (200, 150, 100), (150, 200, 100), (100, 150, 200)
    ]
    
    for field in fields:
        color = colors[(field['field_id'] - 1) % len(colors)]
        overlay[field['mask'] > 0] = color
        
        # Etiquetar con ID
        cx, cy = int(field['centroid'][0]), int(field['centroid'][1])
        cv2.putText(overlay, f"F{field['field_id']}", (cx-10, cy+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Mezclar con transparencia
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
    
    return vis

def create_meeples_visualization(img, meeples_data):
    """Crea visualización de meeples detectados"""
    vis = img.copy()
    
    # Meeples jugador 1 (violeta)
    for meeple in meeples_data[1]:
        cx, cy = int(meeple['centroid'][0]), int(meeple['centroid'][1])
        cv2.circle(vis, (cx, cy), 8, (255, 0, 255), -1)
        cv2.circle(vis, (cx, cy), 9, (255, 255, 255), 2)
        cv2.putText(vis, "1", (cx-4, cy+4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Meeples jugador 2 (negro)
    for meeple in meeples_data[2]:
        cx, cy = int(meeple['centroid'][0]), int(meeple['centroid'][1])
        cv2.circle(vis, (cx, cy), 8, (0, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 9, (255, 255, 255), 2)
        cv2.putText(vis, "2", (cx-4, cy+4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return vis

def create_castles_visualization(img, castles):
    """Crea visualización de castillos (cerrados destacados)"""
    vis = img.copy()
    
    for castle in castles:
        x, y, w, h = castle['bbox']
        
        if castle['closed']:
            # Castillo cerrado - borde verde grueso
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
            label = f"C{castle['castle_id']}"
        else:
            # Castillo abierto - borde rojo
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
            label = f"C{castle['castle_id']}"
        
        cv2.putText(vis, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return vis

def create_final_score_image(img, field_details, scores):
    """Crea imagen final con puntuación superpuesta"""
    vis = img.copy()
    
    # Agregar puntuación por campo
    for field_id, details in field_details.items():
        if 'centroid' in details:
            cx, cy = int(details['centroid'][0]), int(details['centroid'][1])
            
            # Fondo semi-transparente
            text = f"{details['points']}pts"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (cx-tw//2-5, cy-th-5), (cx+tw//2+5, cy+5), (0, 0, 0), -1)
            
            # Texto
            cv2.putText(vis, text, (cx-tw//2, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Agregar totales en la esquina
    y_offset = 30
    cv2.rectangle(vis, (10, 10), (250, 80), (0, 0, 0), -1)
    cv2.putText(vis, f"Jugador 1: {scores[1]} puntos", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(vis, f"Jugador 2: {scores[2]} puntos", (20, y_offset+30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    return vis