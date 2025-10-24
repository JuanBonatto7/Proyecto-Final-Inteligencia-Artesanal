"""
Comparador de fichas con referencias
"""

import cv2
import os
from config import *
from utils import rotate_image, compare_images


class TileMatcher:
    def __init__(self):
        self.references = {}
        self.load_references()
    
    def load_references(self):
        """
        Carga todas las imágenes de referencia
        """
        print("[INFO] Cargando fichas de referencia...")
        
        for tile in TILES:
            path = os.path.join(REFERENCIAS_DIR, f"{tile}.jpg")
            
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    self.references[tile] = img
                    print(f"  ✓ {tile} cargada")
                else:
                    print(f"  ✗ Error al cargar {tile}")
            else:
                print(f"  ✗ No encontrada: {path}")
        
        print(f"[INFO] {len(self.references)} fichas cargadas\n")
    
    def match_tile(self, tile_image):
        """
        Encuentra la mejor coincidencia para una ficha
        Retorna: (letra, rotación, score) o (None, None, 0) si no encuentra
        """
        # Si la ficha está casi vacía o es muy oscura/clara, es vacía
        if self._is_empty_tile(tile_image):
            return (None, None, 0)
        
        best_match = None
        best_rotation = 0
        best_score = 0
        
        # Comparar con cada referencia
        for tile_name, ref_image in self.references.items():
            # Probar cada rotación
            for rotation in ROTATIONS:
                rotated_ref = rotate_image(ref_image, rotation)
                score = compare_images(tile_image, rotated_ref)
                
                if score > best_score:
                    best_score = score
                    best_match = tile_name
                    best_rotation = rotation
        
        # Solo retornar si supera el umbral
        if best_score >= SIMILARITY_THRESHOLD:
            return (best_match, best_rotation, best_score)
        
        return (None, None, best_score)
    
    def _is_empty_tile(self, tile_image):
        """
        Verifica si una ficha está vacía (ej: fuera del tablero)
        """
        gray = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        std_val = gray.std()
        
        # Si tiene muy poca variación, probablemente está vacía
        if std_val < 10:
            return True
        
        return False