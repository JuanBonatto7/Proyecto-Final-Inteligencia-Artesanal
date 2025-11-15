import cv2
import numpy as np

def touches_mask(component_mask, target_mask):
    """Verifica si una componente toca otra máscara (4-conexión)"""
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    dilated = cv2.dilate(component_mask, kernel, iterations=1)
    intersection = cv2.bitwise_and(dilated, target_mask)
    return np.any(intersection > 0)

def assign_meeples_to_fields(meeples_data, fields, field_mask):
    """Asigna cada meeple al campo que toca"""
    field_meeples = {}
    
    # Inicializar contadores para cada campo
    for field in fields:
        field_id = field['field_id']
        field_meeples[field_id] = {1: 0, 2: 0}
    
    # Procesar meeples de cada jugador
    for player in [1, 2]:
        for meeple in meeples_data[player]:
            # Verificar qué campo toca este meeple
            for field in fields:
                if touches_mask(meeple['mask'], field['mask']):
                    field_meeples[field['field_id']][player] += 1
                    break
    
    return field_meeples

def match_castles_fields(castles, fields):
    """Encuentra qué castillos cerrados tocan cada campo"""
    field_castles = {}
    
    for field in fields:
        field_id = field['field_id']
        field_castles[field_id] = []
        
        for castle in castles:
            if castle['closed']:
                # Verificar si este castillo toca el campo
                if touches_mask(castle['mask'], field['mask']):
                    field_castles[field_id].append(castle['castle_id'])
    
    return field_castles

def compute_field_owners(field_meeples):
    """Determina el/los dueño(s) de cada campo"""
    field_owners = {}
    
    for field_id, meeples in field_meeples.items():
        count_1 = meeples[1]
        count_2 = meeples[2]
        
        if count_1 == 0 and count_2 == 0:
            field_owners[field_id] = []  # Sin dueño
        elif count_1 > count_2:
            field_owners[field_id] = [1]  # Jugador 1
        elif count_2 > count_1:
            field_owners[field_id] = [2]  # Jugador 2
        else:  # count_1 == count_2 > 0
            field_owners[field_id] = [1, 2]  # Empate - ambos
    
    return field_owners

def compute_scores(field_owners, field_castles):
    """Calcula los puntos totales para cada jugador"""
    scores = {1: 0, 2: 0}
    field_details = {}
    
    for field_id in field_owners.keys():
        owners = field_owners[field_id]
        closed_castles = field_castles[field_id]
        
        # Calcular puntos: 3 por cada castillo cerrado
        points = len(closed_castles) * 3
        
        # Asignar puntos a los dueños
        for owner in owners:
            scores[owner] += points
        
        # Guardar detalles del campo
        field_details[field_id] = {
            'owners': owners,
            'closed_castles': closed_castles,
            'points': points
        }
    
    return scores, field_details

def score(field_data, meeples_data, castle_data):
    """
    Función principal de puntuación
    
    Args:
        field_data: dict con 'fields' (lista) y 'mask' (numpy array)
        meeples_data: dict {player: [meeples]}
        castle_data: dict con 'castles' (lista) y 'mask' (numpy array)
    
    Returns:
        scores: dict {player: total_points}
        field_details: dict con detalles por campo
    """
    fields = field_data['fields']
    castles = castle_data['castles']
    
    # 1. Asignar meeples a campos
    field_meeples = assign_meeples_to_fields(meeples_data, fields, field_data['mask'])
    
    # 2. Determinar dueños de campos
    field_owners = compute_field_owners(field_meeples)
    
    # 3. Vincular castillos con campos
    field_castles = match_castles_fields(castles, fields)
    
    # 4. Calcular puntuación
    scores, field_details = compute_scores(field_owners, field_castles)
    
    # 5. Agregar información adicional a los detalles
    for field in fields:
        field_id = field['field_id']
        if field_id in field_details:
            field_details[field_id]['meeples'] = field_meeples[field_id]
            field_details[field_id]['centroid'] = field['centroid']
    
    return scores, field_details