#!/usr/bin/env python3
"""
Auto-anotador inteligente de losetas de Carcassonne
Usa las imágenes de referencia para etiquetar automáticamente las losetas detectadas
sin necesidad de intervención humana.

Uso:
    python auto_annotate.py tiles/ referencias/ [--threshold 0.65] [--review]
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass
import sys

from tile_mapping import TileMapper


@dataclass
class MatchResult:
    """Resultado de comparación de una loseta"""
    letter: str
    tile_idx: int
    rotation: int
    confidence: float
    method_scores: Dict[str, float]


class TileComparator:
    """Clase para comparar losetas usando múltiples métricas"""
    
    @staticmethod
    def normalize_tile(img: np.ndarray, size: Tuple[int, int] = (200, 200)) -> np.ndarray:
        """Normaliza una loseta a tamaño estándar"""
        return cv2.resize(img, size)
    
    @staticmethod
    def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula Structural Similarity Index"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        try:
            score = ssim(gray1, gray2)
            return max(0.0, score)  # SSIM puede dar valores negativos
        except:
            return 0.0
    
    @staticmethod
    def compute_histogram_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula correlación de histogramas de color"""
        # Convertir a HSV para mejor comparación de colores
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Calcular histogramas
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        # Normalizar
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Correlación
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0.0, score)
    
    @staticmethod
    def compute_template_matching(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula similitud usando template matching"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Asegurar que img2 no sea más grande que img1
        if gray2.shape[0] > gray1.shape[0] or gray2.shape[1] > gray1.shape[1]:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max(0.0, max_val)
    
    @staticmethod
    def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calcula Mean Squared Error (invertido para que mayor sea mejor)"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        # Normalizar e invertir (menor MSE = mayor similitud)
        max_mse = 255 ** 2
        return 1.0 - (mse / max_mse)
    
    @classmethod
    def compare_tiles(cls, tile1: np.ndarray, tile2: np.ndarray) -> Dict[str, float]:
        """Compara dos losetas usando múltiples métricas"""
        # Normalizar tamaños
        tile1_norm = cls.normalize_tile(tile1)
        tile2_norm = cls.normalize_tile(tile2)
        
        scores = {
            'ssim': cls.compute_ssim(tile1_norm, tile2_norm),
            'histogram': cls.compute_histogram_correlation(tile1_norm, tile2_norm),
            'template': cls.compute_template_matching(tile1_norm, tile2_norm),
            'mse': cls.compute_mse(tile1_norm, tile2_norm)
        }
        
        return scores
    
    @staticmethod
    def aggregate_scores(scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
        """Agrega múltiples métricas en un score final"""
        if weights is None:
            # Pesos por defecto (ajustados empíricamente)
            weights = {
                'ssim': 0.35,
                'histogram': 0.25,
                'template': 0.30,
                'mse': 0.10
            }
        
        total_score = sum(scores[key] * weights[key] for key in scores)
        return total_score


class AutoAnnotator:
    """Anotador automático de losetas"""
    
    def __init__(self, references_dir: str):
        self.references_dir = Path(references_dir)
        self.mapper = TileMapper()
        self.comparator = TileComparator()
        self.reference_tiles = self._load_references()
        
        print(f"✓ Cargadas {len(self.reference_tiles)} losetas de referencia")
    
    def _load_references(self) -> Dict[str, np.ndarray]:
        """Carga las imágenes de referencia"""
        refs = {}
        
        for idx in range(self.mapper.get_num_types()):
            ref_path = self.references_dir / f'tile_type_{idx}.png'
            
            if ref_path.exists():
                img = cv2.imread(str(ref_path))
                if img is not None:
                    letter = self.mapper.idx_to_letter(idx)
                    refs[letter] = img
        
        return refs
    
    def find_best_match(self, tile_img: np.ndarray, 
                       min_confidence: float = 0.0) -> Optional[MatchResult]:
        """
        Encuentra la mejor coincidencia para una loseta.
        Prueba todas las referencias y todas las rotaciones.
        """
        best_match = None
        best_score = -1
        best_letter = None
        best_rotation = 0
        best_method_scores = {}
        
        for letter, ref_img in self.reference_tiles.items():
            # Probar las 4 rotaciones posibles (0°, 90°, 180°, 270°)
            for rotation in range(4):
                # Rotar la loseta a comparar
                rotated_tile = np.rot90(tile_img, rotation)
                
                # Comparar con referencia
                scores = self.comparator.compare_tiles(rotated_tile, ref_img)
                aggregated_score = self.comparator.aggregate_scores(scores)
                
                # Actualizar si es mejor
                if aggregated_score > best_score:
                    best_score = aggregated_score
                    best_letter = letter
                    best_rotation = rotation
                    best_method_scores = scores.copy()
        
        # Verificar umbral mínimo de confianza
        if best_score < min_confidence:
            return None
        
        return MatchResult(
            letter=best_letter,
            tile_idx=self.mapper.letter_to_idx(best_letter),
            rotation=best_rotation,
            confidence=best_score,
            method_scores=best_method_scores
        )
    
    def annotate_tiles(self, tiles_dir: str, 
                      confidence_threshold: float = 0.65,
                      review_low_confidence: bool = False) -> Tuple[List[Dict], List[str]]:
        """
        Anota automáticamente todas las losetas en un directorio.
        
        Args:
            tiles_dir: Directorio con las losetas a anotar
            confidence_threshold: Umbral mínimo de confianza (0-1)
            review_low_confidence: Si True, marca losetas con baja confianza para revisión manual
        
        Returns:
            Tupla con (anotaciones, archivos_para_revisar)
        """
        tiles_path = Path(tiles_dir)
        tile_files = sorted(list(tiles_path.glob('*.png')))
        
        if not tile_files:
            print(f"⚠️  No se encontraron losetas en {tiles_dir}")
            return [], []
        
        print(f"\n{'='*60}")
        print(f"AUTO-ANOTACIÓN DE LOSETAS")
        print(f"{'='*60}")
        print(f"Losetas a procesar: {len(tile_files)}")
        print(f"Umbral de confianza: {confidence_threshold:.2%}")
        print(f"{'='*60}\n")
        
        annotations = []
        files_to_review = []
        
        stats = {
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'failed': 0
        }
        
        for i, tile_file in enumerate(tile_files, 1):
            # Cargar loseta
            tile_img = cv2.imread(str(tile_file))
            
            if tile_img is None:
                print(f"✗ Error cargando: {tile_file.name}")
                stats['failed'] += 1
                continue
            
            # Buscar mejor coincidencia
            match = self.find_best_match(tile_img, min_confidence=confidence_threshold)
            
            if match is None:
                print(f"[{i}/{len(tile_files)}] ⚠️  {tile_file.name}: Sin coincidencia confiable")
                files_to_review.append(str(tile_file))
                stats['failed'] += 1
                continue
            
            # Categorizar por confianza
            if match.confidence >= 0.85:
                status = "✓"
                confidence_level = "ALTA"
                stats['high_confidence'] += 1
            elif match.confidence >= 0.75:
                status = "○"
                confidence_level = "MEDIA"
                stats['medium_confidence'] += 1
            else:
                status = "?"
                confidence_level = "BAJA"
                stats['low_confidence'] += 1
                if review_low_confidence:
                    files_to_review.append(str(tile_file))
            
            # Crear anotación
            annotation = {
                'image_path': str(tile_file),
                'tile_letter': match.letter,
                'tile_type': match.tile_idx,
                'rotation': match.rotation,
                'confidence': float(match.confidence),
                'method_scores': {k: float(v) for k, v in match.method_scores.items()},
                'has_meeple': False,  # Por defecto sin ficha
                'meeple_position': -1,
                'meeple_color': 'none',
                'auto_annotated': True
            }
            
            annotations.append(annotation)
            
            # Mostrar progreso
            rotation_deg = match.rotation * 90
            print(f"[{i}/{len(tile_files)}] {status} {tile_file.name}: "
                  f"{match.letter} (rot {rotation_deg}°) - "
                  f"Confianza: {match.confidence:.2%} ({confidence_level})")
        
        # Resumen
        print(f"\n{'='*60}")
        print(f"RESUMEN DE AUTO-ANOTACIÓN")
        print(f"{'='*60}")
        print(f"✓ Alta confianza (≥85%):    {stats['high_confidence']}")
        print(f"○ Media confianza (75-85%): {stats['medium_confidence']}")
        print(f"? Baja confianza (65-75%):  {stats['low_confidence']}")
        print(f"✗ Fallidas (<65%):          {stats['failed']}")
        print(f"{'='*60}")
        print(f"Total anotadas: {len(annotations)}/{len(tile_files)}")
        
        if files_to_review:
            print(f"\n⚠️  {len(files_to_review)} losetas marcadas para revisión manual")
        
        return annotations, files_to_review
    
    def export_annotations(self, annotations: List[Dict], output_file: str = 'auto_annotations.json'):
        """Exporta las anotaciones a JSON"""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Anotaciones guardadas en: {output_path}")
        print(f"  Total de anotaciones: {len(annotations)}")
    
    def export_review_list(self, files_to_review: List[str], output_file: str = 'documentacion/review_list.txt'):
        """Exporta lista de archivos para revisión manual"""
        if not files_to_review:
            return
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# LOSETAS PARA REVISIÓN MANUAL\n")
            f.write(f"# Total: {len(files_to_review)}\n\n")
            for file_path in files_to_review:
                f.write(f"{file_path}\n")
        
        print(f"✓ Lista de revisión guardada en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Auto-anotador inteligente de losetas de Carcassonne',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Anotación básica
  python auto_annotate.py tiles/ referencias/
  
  # Con umbral personalizado
  python auto_annotate.py tiles/ referencias/ --threshold 0.70
  
  # Marcar casos de baja confianza para revisión
  python auto_annotate.py tiles/ referencias/ --review
  
  # Especificar archivo de salida
  python auto_annotate.py tiles/ referencias/ --output annotations.json
        """
    )
    
    parser.add_argument('tiles_dir', help='Directorio con las losetas a anotar')
    parser.add_argument('references_dir', help='Directorio con las losetas de referencia')
    parser.add_argument('--threshold', '-t', type=float, default=0.65,
                       help='Umbral mínimo de confianza (0.0-1.0, default: 0.65)')
    parser.add_argument('--review', '-r', action='store_true',
                       help='Marcar losetas de confianza media/baja para revisión manual')
    parser.add_argument('--output', '-o', default='auto_annotations.json',
                       help='Archivo de salida (default: auto_annotations.json)')
    
    args = parser.parse_args()
    
    # Validar paths
    if not Path(args.tiles_dir).exists():
        print(f"✗ Error: No existe el directorio de losetas: {args.tiles_dir}")
        sys.exit(1)
    
    if not Path(args.references_dir).exists():
        print(f"✗ Error: No existe el directorio de referencias: {args.references_dir}")
        sys.exit(1)
    
    # Validar threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"✗ Error: El threshold debe estar entre 0.0 y 1.0")
        sys.exit(1)
    
    try:
        # Crear auto-anotador
        annotator = AutoAnnotator(args.references_dir)
        
        # Anotar losetas
        annotations, files_to_review = annotator.annotate_tiles(
            args.tiles_dir,
            confidence_threshold=args.threshold,
            review_low_confidence=args.review
        )
        
        # Exportar resultados
        if annotations:
            annotator.export_annotations(annotations, args.output)
        
        if files_to_review:
            annotator.export_review_list(files_to_review)
        
        # Mensaje final
        if not annotations:
            print("\n⚠️  No se pudo anotar ninguna loseta automáticamente")
            print("   Verifica que las imágenes de referencia sean correctas")
            return 1
        
        print(f"\n{'='*60}")
        print("✓ PROCESO COMPLETADO")
        print(f"{'='*60}")
        
        if files_to_review:
            print("\nPasos siguientes:")
            print("1. Revisa manualmente las losetas en 'documentacion/review_list.txt'")
            print("2. Usa annotation_tool_letters.py para corregir/completar")
            print("3. Combina las anotaciones automáticas con las manuales")
        else:
            print("\n✓ Todas las losetas fueron anotadas con alta confianza")
            print(f"  Puedes usar directamente: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error durante la auto-anotación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
