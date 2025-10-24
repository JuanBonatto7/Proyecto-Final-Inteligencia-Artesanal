"""
Pipeline principal para procesamiento secuencial de imágenes.
"""

from typing import List, Dict, Any
import time
import cv2
from .pipeline_step import PipelineStep


class Pipeline:
    """
    Pipeline que ejecuta una secuencia de pasos de procesamiento.
    
    Permite agregar pasos dinámicamente y ejecutarlos en orden,
    pasando los resultados de un paso al siguiente.
    """
    
    def __init__(self, name: str = "Pipeline"):
        """
        Inicializa el pipeline.
        
        Args:
            name: Nombre descriptivo del pipeline
        """
        self.name = name
        self.steps: List[PipelineStep] = []
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """
        Agrega un paso al pipeline.
        
        Args:
            step: Instancia de PipelineStep a agregar
            
        Returns:
            Self para permitir chaining
        """
        self.steps.append(step)
        return self
    
    def run(self, inputs: Dict[str, Any], visualize: bool = False) -> Dict[str, Any]:
        """
        Ejecuta todos los pasos del pipeline en secuencia.
        
        Args:
            inputs: Diccionario con datos de entrada iniciales.
                   Debe contener al menos {'img': image_array}
            visualize: Si True, muestra visualizaciones de cada paso
            
        Returns:
            Diccionario con todos los resultados del pipeline
            
        Raises:
            ValueError: Si inputs no contiene 'img'
            Exception: Si algún paso falla durante el procesamiento
        """
        if 'img' not in inputs:
            raise ValueError("El diccionario de entrada debe contener 'img'")
        
        print(f"\n{'='*60}")
        print(f"  Iniciando {self.name}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        results = inputs.copy()
        visualization_images = []
        
        for idx, step in enumerate(self.steps, 1):
            print(f"[{idx}/{len(self.steps)}] Ejecutando: {step.name}...", end=" ")
            
            step_start = time.time()
            
            try:
                results = step.process(results)
                step_time = time.time() - step_start
                
                print(f"✓ ({step_time:.3f}s)")
                
                if visualize:
                    vis_image = step.get_visualization_image(results)
                    if vis_image is not None:
                        visualization_images.append((step.name, vis_image))
                        
            except Exception as e:
                print(f"✗ ERROR")
                print(f"Error en paso '{step.name}': {str(e)}")
                raise
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"  Pipeline completado en {total_time:.3f} segundos")
        print(f"{'='*60}\n")
        
        if visualize and visualization_images:
            self._show_visualizations(visualization_images)
        
        return results
    
    def _show_visualizations(self, images: List[tuple]) -> None:
        """
        Muestra las imágenes de visualización de cada paso.
        
        Args:
            images: Lista de tuplas (nombre, imagen)
        """
        for name, image in images:
            cv2.imshow(f"{self.name} - {name}", image)
        
        print("\nPresiona cualquier tecla para cerrar las visualizaciones...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def __len__(self) -> int:
        """Retorna el número de pasos en el pipeline."""
        return len(self.steps)
    
    def __str__(self) -> str:
        """Retorna descripción del pipeline."""
        steps_names = [step.name for step in self.steps]
        return f"{self.name} con {len(self.steps)} pasos: {steps_names}"