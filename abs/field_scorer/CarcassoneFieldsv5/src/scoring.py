"""
Cálculo de puntuación de campos.
Solo cuenta castillos COMPLETOS para puntos.
Castillos incompletos delimitan pero no puntúan.
"""
from typing import Dict, List, Tuple
from src.field_detector import Field
import numpy as np
from scipy import ndimage
import cv2
import os


class FieldScorer:
    """Calcula puntuación de campos."""
    
    def __init__(self, castle_mask: np.ndarray, castle_analyzer=None):
        """
        Inicializa el calculador de puntos.
        
        Args:
            castle_mask: Máscara de TODOS los castillos
            castle_analyzer: Analizador de castillos (opcional)
        """
        self.castle_mask = castle_mask
        self.castle_analyzer = castle_analyzer
        
        # Etiquetar castillos individuales
        self.labeled_castles, self.num_castles = ndimage.label(
            castle_mask, 
            structure=np.ones((3, 3), dtype=int)
        )
        
        # Si tenemos analizador, crear etiquetado de solo castillos completos
        if castle_analyzer:
            complete_mask = castle_analyzer.get_complete_castles_mask()
            self.labeled_complete_castles, self.num_complete_castles = ndimage.label(
                complete_mask,
                structure=np.ones((3, 3), dtype=int)
            )
        else:
            # Fallback: asumir todos completos si no hay analizador
            self.labeled_complete_castles = self.labeled_castles
            self.num_complete_castles = self.num_castles
    
    def count_adjacent_castles(self, field: Field, only_complete: bool = True) -> int:
        """
        Cuenta castillos adyacentes o dentro de un campo.
        
        Args:
            field: Campo a analizar
            only_complete: Si True, solo cuenta castillos completos
            
        Returns:
            Número de castillos adyacentes únicos
        """
        # Expandir el campo para detectar castillos en el borde o dentro
        kernel = np.ones((7, 7), dtype=np.uint8)
        expanded_field = ndimage.binary_dilation(field.pixels, structure=kernel, iterations=3)
        
        # Elegir qué castillos contar
        if only_complete and self.castle_analyzer:
            # Solo castillos COMPLETOS cuentan para puntos
            labeled_to_use = self.labeled_complete_castles
            castles_in_or_near_field = expanded_field & self.castle_analyzer.get_complete_castles_mask()
        else:
            # Todos los castillos (usado para barreras)
            labeled_to_use = self.labeled_castles
            castles_in_or_near_field = expanded_field & self.castle_mask
        
        if not np.any(castles_in_or_near_field):
            return 0
        
        # Identificar qué castillos únicos están presentes
        unique_castle_ids = set()
        
        y_coords, x_coords = np.where(castles_in_or_near_field)
        
        for y, x in zip(y_coords, x_coords):
            castle_id = labeled_to_use[y, x]
            if castle_id > 0:  # 0 es el fondo
                unique_castle_ids.add(castle_id)
        
        return len(unique_castle_ids)
    
    def get_castle_ids_for_field(self, field: Field, only_complete: bool = True) -> List[int]:
        """
        Obtiene los IDs de castillos adyacentes a un campo.
        
        Args:
            field: Campo a analizar
            only_complete: Si True, solo castillos completos
            
        Returns:
            Lista de IDs de castillos
        """
        kernel = np.ones((7, 7), dtype=np.uint8)
        expanded_field = ndimage.binary_dilation(field.pixels, structure=kernel, iterations=3)
        
        if only_complete and self.castle_analyzer:
            labeled_to_use = self.labeled_complete_castles
            castles_in_or_near_field = expanded_field & self.castle_analyzer.get_complete_castles_mask()
        else:
            labeled_to_use = self.labeled_castles
            castles_in_or_near_field = expanded_field & self.castle_mask
        
        if not np.any(castles_in_or_near_field):
            return []
        
        unique_castle_ids = set()
        y_coords, x_coords = np.where(castles_in_or_near_field)
        
        for y, x in zip(y_coords, x_coords):
            castle_id = labeled_to_use[y, x]
            if castle_id > 0:
                unique_castle_ids.add(castle_id)
        
        return sorted(list(unique_castle_ids))
    
    def determine_owner(self, field: Field) -> Tuple[str, bool]:
        """
        Determina el dueño de un campo.
        El jugador con MÁS meeples es el dueño.
        Si varios jugadores tienen la misma cantidad máxima, hay empate.
        
        Args:
            field: Campo a analizar
            
        Returns:
            (owner, is_tie): Tupla con el dueño y si hay empate
        """
        if not field.meeples or all(count == 0 for count in field.meeples.values()):
            return None, False
        
        # Encontrar la cantidad máxima de meeples
        max_count = max(field.meeples.values())
        
        # Encontrar todos los jugadores con esa cantidad máxima
        owners = [player for player, count in field.meeples.items() if count == max_count]
        
        # Si hay más de un jugador con el máximo, hay empate
        if len(owners) > 1:
            return 'TIE', True
        
        # Si solo hay uno con el máximo, es el dueño
        return owners[0], False
    
    def calculate_field_score(self, field: Field) -> int:
        """
        Calcula puntos de un campo.
        SOLO cuenta castillos COMPLETOS (3 puntos c/u).
        
        Args:
            field: Campo a puntuar
            
        Returns:
            Puntos del campo
        """
        num_castles = self.count_adjacent_castles(field, only_complete=True)
        return num_castles * 3
    
    def calculate_all_scores(
        self, 
        fields: List[Field]
    ) -> Dict[str, Dict]:
        """
        Calcula puntuación para todos los campos.
        
        Args:
            fields: Lista de campos
            
        Returns:
            Diccionario con información de puntuación por campo
        """
        results = {}
        
        for field in fields:
            owner, is_tie = self.determine_owner(field)
            
            # Castillos COMPLETOS para puntos
            complete_castles = self.count_adjacent_castles(field, only_complete=True)
            score = complete_castles * 3
            
            # Total de castillos (completos + incompletos) para info
            total_castles = self.count_adjacent_castles(field, only_complete=False)
            
            results[field.id] = {
                'owner': owner,
                'is_tie': is_tie,
                'score': score,
                'meeples': field.meeples.copy(),
                'castles': complete_castles,  # Solo completos para puntos
                'castles_complete': complete_castles,
                'castles_incomplete': total_castles - complete_castles,
                'area': field.area
            }
        
        return results
    
    def calculate_player_totals(
        self, 
        field_results: Dict[str, Dict]
    ) -> Dict[str, int]:
        """
        Calcula puntos totales por jugador.
        En caso de empate, TODOS los jugadores empatados obtienen los puntos completos.
        
        Args:
            field_results: Resultados por campo
            
        Returns:
            Puntos totales por jugador
        """
        # Inicializar SIEMPRE ambos jugadores
        totals = {
            'MEEPLE_1': 0,
            'MEEPLE_2': 0,
        }
        
        for field_data in field_results.values():
            is_tie = field_data['is_tie']
            score = field_data['score']
            meeples = field_data['meeples']
            
            if score > 0:
                if is_tie:
                    # En caso de empate, todos los jugadores con meeples obtienen puntos completos
                    for player, count in meeples.items():
                        if count > 0:
                            totals[player] += score
                else:
                    # Solo el dueño obtiene puntos
                    owner = field_data['owner']
                    if owner and owner != 'TIE':
                        totals[owner] += score
        
        return totals
    
    def save_castle_details(self, fields: List[Field], field_results: Dict, output_folder: str, original_image: np.ndarray):
        """
        Guarda información detallada de los castillos por campo.
        
        Args:
            fields: Lista de campos
            field_results: Resultados de puntuación
            output_folder: Carpeta donde guardar
            original_image: Imagen original del tablero
        """
        details_file = os.path.join(output_folder, "castillos_por_campo.txt")
        
        with open(details_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DETALLE DE CASTILLOS POR CAMPO\n")
            f.write("=" * 80 + "\n\n")
            
            for field in fields:
                f.write(f"\n{'='*80}\n")
                f.write(f"CAMPO {field.id}\n")
                f.write(f"{'='*80}\n")
                
                # Castillos completos
                complete_ids = self.get_castle_ids_for_field(field, only_complete=True)
                f.write(f"\nCASTILLOS COMPLETOS (cuentan para puntos): {len(complete_ids)}\n")
                if complete_ids:
                    f.write(f"  IDs: {complete_ids}\n")
                    for castle_id in complete_ids:
                        castle_mask = (self.labeled_complete_castles == castle_id)
                        area = castle_mask.sum()
                        f.write(f"    - Castillo #{castle_id}: {area} pixels\n")
                else:
                    f.write("  Ninguno\n")
                
                # Castillos incompletos
                all_ids = self.get_castle_ids_for_field(field, only_complete=False)
                incomplete_ids = [cid for cid in all_ids if cid not in complete_ids]
                f.write(f"\nCASTILLOS INCOMPLETOS (NO cuentan para puntos): {len(incomplete_ids)}\n")
                if incomplete_ids:
                    f.write(f"  IDs: {incomplete_ids}\n")
                    for castle_id in incomplete_ids:
                        castle_mask = (self.labeled_castles == castle_id)
                        area = castle_mask.sum()
                        f.write(f"    - Castillo #{castle_id}: {area} pixels (toca borde)\n")
                else:
                    f.write("  Ninguno\n")
                
                # Puntuación
                f.write(f"\nPUNTUACION:\n")
                f.write(f"  Castillos completos: {len(complete_ids)}\n")
                f.write(f"  Puntos por castillo: 3\n")
                f.write(f"  TOTAL: {len(complete_ids) * 3} puntos\n")
                
                # Info del campo
                result = field_results.get(field.id, {})
                f.write(f"\nINFO DEL CAMPO:\n")
                f.write(f"  Área: {field.area} pixels\n")
                f.write(f"  Meeples: {field.meeples}\n")
                f.write(f"  Dueño: {result.get('owner', 'Ninguno')}\n")
                
                # Generar imagen individual del campo con castillos
                self._save_field_castle_image(field, complete_ids, incomplete_ids, 
                                              original_image, output_folder)
            
            f.write(f"\n\n{'='*80}\n")
            f.write("RESUMEN GENERAL\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total de campos analizados: {len(fields)}\n")
            if self.castle_analyzer:
                stats = self.castle_analyzer.get_castle_statistics()
                f.write(f"Total de castillos en el tablero: {stats['total_castles']}\n")
                f.write(f"  - Completos: {stats['complete_castles']}\n")
                f.write(f"  - Incompletos: {stats['incomplete_castles']}\n")
        
        print(f"   [OK] Detalles guardados en: {details_file}")
    
    def _save_field_castle_image(self, field: Field, complete_ids: List[int], 
                                  incomplete_ids: List[int], original_image: np.ndarray, 
                                  output_folder: str):
        """
        Guarda imagen individual de un campo mostrando sus castillos.
        
        Args:
            field: Campo a visualizar
            complete_ids: IDs de castillos completos
            incomplete_ids: IDs de castillos incompletos
            original_image: Imagen original
            output_folder: Carpeta de salida
        """
        img = original_image.copy()
        
        # Resaltar el campo en amarillo transparente
        overlay = img.copy()
        overlay[field.pixels] = [255, 255, 150]
        img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        
        # Dibujar castillos completos en verde
        for castle_id in complete_ids:
            castle_mask = (self.labeled_complete_castles == castle_id)
            contours, _ = cv2.findContours(
                castle_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            
            # Etiquetar castillo
            y, x = np.where(castle_mask)
            if len(y) > 0:
                cy, cx = int(y.mean()), int(x.mean())
                cv2.putText(img, f"C{castle_id}", (cx-10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Dibujar castillos incompletos en rojo
        for castle_id in incomplete_ids:
            castle_mask = (self.labeled_castles == castle_id)
            contours, _ = cv2.findContours(
                castle_mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(img, contours, -1, (255, 0, 0), 3)
            
            # Etiquetar castillo
            y, x = np.where(castle_mask)
            if len(y) > 0:
                cy, cx = int(y.mean()), int(x.mean())
                cv2.putText(img, f"I{castle_id}", (cx-10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Guardar imagen
        filename = os.path.join(output_folder, f"campo_{field.id}_castillos.png")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))