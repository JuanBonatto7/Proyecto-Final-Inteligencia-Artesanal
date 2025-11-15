import cv2
import numpy as np
import os
from image_processing import (
    segment_colors, 
    find_components, 
    detect_meeples,
    find_fields, 
    find_castles,
    create_visualization_masks,
    create_fields_labeled,
    create_meeples_visualization,
    create_castles_visualization,
    create_final_score_image
)
from scoring import score

# Paleta de colores
COLORS = {
    'FIELD': (34, 177, 76),           # Verde
    'CASTLE': (255, 127, 39),         # Naranja
    'ROAD': (63, 72, 204),            # Azul
    'VERTEX': (237, 28, 36),          # Rojo
    'MEEPLE_1': (163, 73, 164),       # Violeta
    'MEEPLE_2': (0, 0, 0),            # Negro
    'TILE_BG': (255, 255, 255)        # Blanco
}

def main(image_path):
    # Crear carpeta de salida
    os.makedirs('outputs', exist_ok=True)
    
    # Cargar imagen
    print(f"Cargando imagen: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return
    
    print(f"Dimensiones: {img.shape}")
    
    # 1. Segmentar colores
    print("\n=== PASO 1: Segmentación de colores ===")
    masks = segment_colors(img, COLORS)
    
    # Guardar visualización de máscaras
    mask_vis = create_visualization_masks(masks, img.shape[:2])
    cv2.imwrite('outputs/01_masks.png', mask_vis)
    print("✓ Máscaras guardadas en outputs/01_masks.png")
    
    # 2. Encontrar campos
    print("\n=== PASO 2: Detectando campos ===")
    fields = find_fields(masks['FIELD'])
    print(f"✓ {len(fields)} campos encontrados")
    
    # Guardar campos etiquetados
    fields_vis = create_fields_labeled(img, fields)
    cv2.imwrite('outputs/02_fields_labeled.png', fields_vis)
    print("✓ Campos etiquetados guardados en outputs/02_fields_labeled.png")
    
    # 3. Encontrar castillos
    print("\n=== PASO 3: Detectando castillos ===")
    castles = find_castles(masks['CASTLE'], masks['TILE_BG'], img.shape[:2])
    closed_count = sum(1 for c in castles if c['closed'])
    print(f"✓ {len(castles)} castillos encontrados ({closed_count} cerrados)")
    
    # Guardar castillos
    castles_vis = create_castles_visualization(img, castles)
    cv2.imwrite('outputs/04_castles_closed.png', castles_vis)
    print("✓ Castillos guardados en outputs/04_castles_closed.png")
    
    # 4. Detectar meeples
    print("\n=== PASO 4: Detectando meeples ===")
    meeples_data = detect_meeples(masks, img.shape[:2])
    print(f"✓ Jugador 1: {len(meeples_data[1])} meeples")
    print(f"✓ Jugador 2: {len(meeples_data[2])} meeples")
    
    # Guardar meeples
    meeples_vis = create_meeples_visualization(img, meeples_data)
    cv2.imwrite('outputs/03_meeples_detected.png', meeples_vis)
    print("✓ Meeples guardados en outputs/03_meeples_detected.png")
    
    # 5. Calcular puntuación
    print("\n=== PASO 5: Calculando puntuación ===")
    field_data = {'fields': fields, 'mask': masks['FIELD']}
    castle_data = {'castles': castles, 'mask': masks['CASTLE']}
    
    scores, field_details = score(field_data, meeples_data, castle_data)
    
    # Mostrar resultados detallados
    print("\n" + "="*60)
    print("RESULTADOS POR CAMPO")
    print("="*60)
    
    for field_id, details in sorted(field_details.items()):
        print(f"\nCampo {field_id}:")
        
        if details['owners']:
            owners_str = ", ".join([f"Jugador {o}" for o in details['owners']])
            print(f"  Dueño(s): {owners_str}")
        else:
            print(f"  Dueño(s): Ninguno")
        
        print(f"  Meeples - J1: {details['meeples'][1]}, J2: {details['meeples'][2]}")
        
        if details['closed_castles']:
            print(f"  Castillos cerrados: {details['closed_castles']}")
        else:
            print(f"  Castillos cerrados: Ninguno")
        
        print(f"  Puntos: {details['points']}")
        
        if details['owners']:
            recipients = ", ".join([f"Jugador {o}" for o in details['owners']])
            print(f"  → {recipients} recibe(n) {details['points']} puntos")
    
    # Mostrar totales
    print("\n" + "="*60)
    print("PUNTUACIÓN FINAL")
    print("="*60)
    print(f"Jugador 1: {scores[1]} puntos")
    print(f"Jugador 2: {scores[2]} puntos")
    print("="*60)
    
    # Crear imagen final con puntuación
    final_vis = create_final_score_image(img, field_details, scores)
    cv2.imwrite('outputs/final_score.png', final_vis)
    print("\n✓ Imagen final guardada en outputs/final_score.png")
    
    print("\n¡Análisis completado!")

if __name__ == "__main__":
    # Cambiar esta ruta por tu imagen
    image_path = "carcassonne_board.png"
    main(image_path)