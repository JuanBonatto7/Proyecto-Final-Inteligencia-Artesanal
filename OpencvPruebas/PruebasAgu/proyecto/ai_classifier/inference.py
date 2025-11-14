"""
Pipeline de Inferencia para Clasificación de Losetas

Este módulo permite hacer predicciones sobre nuevas imágenes de losetas.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from model import CarcassonneCNN, create_model
from dataset import CarcassonneDataset


class TileClassifier:
    """Clasificador de losetas de Carcassonne."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        image_size: int = 224
    ):
        """
        Inicializa el clasificador.
        
        Args:
            model_path: Ruta al modelo entrenado (.pth)
            device: Dispositivo (cuda/cpu)
            image_size: Tamaño de entrada del modelo
        """
        self.image_size = image_size
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Cargar modelo
        print(f"Cargando modelo desde {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Modelo cargado en {self.device}")
        
        # Transformaciones (mismas que en entrenamiento)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocesa una imagen para el modelo.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Tensor con shape (1, 3, H, W)
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Agregar dimensión de batch
    
    def predict_single(
        self,
        image_path: str,
        return_confidence: bool = True
    ) -> Dict:
        """
        Hace predicción sobre una sola imagen.
        
        Args:
            image_path: Ruta a la imagen
            return_confidence: Si retornar confianzas
            
        Returns:
            Diccionario con predicciones
        """
        # Preprocesar
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Predecir
        with torch.no_grad():
            predictions = self.model.predict(image_tensor)
        
        # Extraer resultados
        tile_type_idx = predictions['tile_type'].item()
        rotation = predictions['rotation'].item()
        has_meeple = predictions['meeple_presence'].item() == 1
        meeple_position = predictions['meeple_position'].item()
        meeple_color_idx = predictions['meeple_color'].item()
        
        # Convertir índice de color a string (0=blue, 1=black, -1=sin meeple)
        meeple_color = None
        if has_meeple and meeple_color_idx != -1:
            meeple_color = 'blue' if meeple_color_idx == 0 else 'black'
        
        result = {
            'image_path': image_path,
            'tile_type': tile_type_idx,
            'tile_letter': CarcassonneDataset.IDX_TO_LETTER[tile_type_idx],
            'rotation': rotation,
            'has_meeple': has_meeple,
            'meeple_position': meeple_position,
            'meeple_color': meeple_color
        }
        
        if return_confidence:
            result['confidence'] = {
                'tile_type': predictions['confidence']['tile_type'].item(),
                'rotation': predictions['confidence']['rotation'].item(),
                'meeple_presence': predictions['confidence']['meeple_presence'].item(),
                'meeple_position': predictions['confidence']['meeple_position'].item(),
                'meeple_color': predictions['confidence']['meeple_color'].item()
            }
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Hace predicciones sobre múltiples imágenes.
        
        Args:
            image_paths: Lista de rutas a imágenes
            batch_size: Tamaño del batch
            
        Returns:
            Lista de diccionarios con predicciones
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            # Preprocesar batch
            for path in batch_paths:
                tensor = self.preprocess_image(path)
                batch_tensors.append(tensor)
            
            # Crear batch
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Predecir
            with torch.no_grad():
                predictions = self.model.predict(batch)
            
            # Procesar resultados
            for j, path in enumerate(batch_paths):
                tile_type_idx = predictions['tile_type'][j].item()
                rotation = predictions['rotation'][j].item()
                has_meeple = predictions['meeple_presence'][j].item() == 1
                meeple_position = predictions['meeple_position'][j].item()
                meeple_color_idx = predictions['meeple_color'][j].item()
                
                # Convertir índice de color a string
                meeple_color = None
                if has_meeple and meeple_color_idx != -1:
                    meeple_color = 'blue' if meeple_color_idx == 0 else 'black'
                
                result = {
                    'image_path': path,
                    'tile_type': tile_type_idx,
                    'tile_letter': CarcassonneDataset.IDX_TO_LETTER[tile_type_idx],
                    'rotation': rotation,
                    'has_meeple': has_meeple,
                    'meeple_position': meeple_position,
                    'meeple_color': meeple_color,
                    'confidence': {
                        'tile_type': predictions['confidence']['tile_type'][j].item(),
                        'rotation': predictions['confidence']['rotation'][j].item(),
                        'meeple_presence': predictions['confidence']['meeple_presence'][j].item(),
                        'meeple_position': predictions['confidence']['meeple_position'][j].item(),
                        'meeple_color': predictions['confidence']['meeple_color'][j].item()
                    }
                }
                
                results.append(result)
        
        return results
    
    def predict_directory(
        self,
        directory: str,
        output_file: Optional[str] = None,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Hace predicciones sobre todas las imágenes en un directorio.
        
        Args:
            directory: Directorio con imágenes
            output_file: Archivo donde guardar resultados (opcional)
            batch_size: Tamaño del batch
            
        Returns:
            Lista de predicciones
        """
        import glob
        
        # Buscar imágenes
        patterns = ['*.png', '*.jpg', '*.jpeg']
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(directory, pattern)))
        
        print(f"Encontradas {len(image_paths)} imágenes en {directory}")
        
        # Predecir
        results = self.predict_batch(image_paths, batch_size=batch_size)
        
        # Guardar si se especifica
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Resultados guardados en {output_file}")
        
        return results
    
    def visualize_prediction(
        self,
        image_path: str,
        prediction: Optional[Dict] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualiza la predicción sobre la imagen.
        
        Args:
            image_path: Ruta a la imagen
            prediction: Predicción (si es None, se calcula)
            save_path: Donde guardar la visualización
        """
        # Obtener predicción si no se proporciona
        if prediction is None:
            prediction = self.predict_single(image_path)
        
        # Cargar imagen
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Crear panel para información
        panel_height = 200
        canvas = np.zeros((h + panel_height, w, 3), dtype=np.uint8)
        canvas[:h, :] = image
        canvas[h:, :] = (40, 40, 40)
        
        # Agregar texto
        y_offset = h + 30
        line_height = 35
        
        def draw_text(text, y, color=(255, 255, 255)):
            cv2.putText(canvas, text, (20, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Información de predicción
        tile_text = f"Tipo: {prediction['tile_letter']}"
        rotation_text = f"Rotacion: {prediction['rotation']} ({prediction['rotation'] * 90} grados)"
        meeple_text = f"Meeple: {'SI' if prediction['has_meeple'] else 'NO'}"
        
        draw_text(tile_text, y_offset, (0, 255, 255))
        y_offset += line_height
        draw_text(rotation_text, y_offset, (0, 255, 0))
        y_offset += line_height
        
        meeple_color = (0, 255, 0) if prediction['has_meeple'] else (0, 0, 255)
        draw_text(meeple_text, y_offset, meeple_color)
        
        if prediction['has_meeple']:
            y_offset += line_height
            pos_text = f"Posicion: {prediction['meeple_position']}"
            draw_text(pos_text, y_offset, (255, 255, 0))
            
            if prediction.get('meeple_color'):
                y_offset += line_height
                color_text = f"Color: {prediction['meeple_color'].upper()}"
                color_rgb = (255, 200, 100) if prediction['meeple_color'] == 'blue' else (200, 200, 200)
                draw_text(color_text, y_offset, color_rgb)
        
        # Mostrar o guardar
        if save_path:
            cv2.imwrite(save_path, canvas)
            print(f"✓ Visualización guardada en {save_path}")
        else:
            cv2.imshow("Prediccion", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return canvas


def classify_tiles_from_detector(
    detector_tiles_dir: str,
    model_path: str,
    output_json: str = 'predictions.json'
):
    """
    Clasifica losetas extraídas por el detector.
    
    Args:
        detector_tiles_dir: Directorio con losetas del detector
        model_path: Ruta al modelo entrenado
        output_json: Archivo de salida con predicciones
    """
    print("\n" + "="*70)
    print("CLASIFICACIÓN DE LOSETAS")
    print("="*70)
    
    # Crear clasificador
    classifier = TileClassifier(model_path)
    
    # Predecir en todo el directorio
    results = classifier.predict_directory(
        directory=detector_tiles_dir,
        output_file=output_json,
        batch_size=32
    )
    
    # Mostrar resumen
    print("\n" + "="*70)
    print("RESUMEN DE PREDICCIONES")
    print("="*70)
    print(f"Total de losetas clasificadas: {len(results)}")
    
    # Contar tipos
    tile_counts = {}
    for result in results:
        letter = result['tile_letter']
        tile_counts[letter] = tile_counts.get(letter, 0) + 1
    
    print("\nDistribución de tipos:")
    for letter in sorted(tile_counts.keys()):
        print(f"  {letter}: {tile_counts[letter]}")
    
    # Contar rotaciones
    rotation_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for result in results:
        rotation_counts[result['rotation']] += 1
    
    print("\nDistribución de rotaciones:")
    for rot, count in rotation_counts.items():
        print(f"  {rot * 90}°: {count}")
    
    # Contar meeples
    meeple_count = sum(1 for r in results if r['has_meeple'])
    print(f"\nLosetas con meeple: {meeple_count}/{len(results)}")
    
    print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Clasificar losetas de Carcassonne')
    parser.add_argument('model_path', type=str, help='Ruta al modelo entrenado (.pth)')
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: single (clasificar una imagen)
    single_parser = subparsers.add_parser('single', help='Clasificar una sola imagen')
    single_parser.add_argument('image', type=str, help='Ruta a la imagen')
    single_parser.add_argument('--visualize', action='store_true', help='Mostrar visualización')
    single_parser.add_argument('--save', type=str, help='Guardar visualización en archivo')
    
    # Comando: batch (clasificar directorio)
    batch_parser = subparsers.add_parser('batch', help='Clasificar directorio de imágenes')
    batch_parser.add_argument('directory', type=str, help='Directorio con imágenes')
    batch_parser.add_argument('--output', type=str, default='predictions.json',
                             help='Archivo de salida')
    batch_parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        classifier = TileClassifier(args.model_path)
        prediction = classifier.predict_single(args.image)
        
        print("\n" + "="*70)
        print("PREDICCIÓN")
        print("="*70)
        print(f"Imagen: {args.image}")
        print(f"Tipo: {prediction['tile_letter']}")
        print(f"Rotación: {prediction['rotation']} ({prediction['rotation'] * 90}°)")
        print(f"Meeple: {'SI' if prediction['has_meeple'] else 'NO'}")
        if prediction['has_meeple']:
            print(f"Posición: {prediction['meeple_position']}")
            if prediction.get('meeple_color'):
                print(f"Color: {prediction['meeple_color'].upper()}")
        print("\nConfianzas:")
        for key, value in prediction['confidence'].items():
            print(f"  {key}: {value:.4f}")
        print("="*70)
        
        if args.visualize or args.save:
            classifier.visualize_prediction(args.image, prediction, args.save)
    
    elif args.command == 'batch':
        classify_tiles_from_detector(
            detector_tiles_dir=args.directory,
            model_path=args.model_path,
            output_json=args.output
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
