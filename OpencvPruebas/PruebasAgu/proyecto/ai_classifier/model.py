"""
Modelo de Red Neuronal Convolucional Multi-tarea para Clasificación de Losetas de Carcassonne

Este módulo implementa una arquitectura CNN que clasifica simultáneamente:
- Tipo de loseta (A-X + BLANCO, 25 clases)
- Rotación (0-3, 4 clases)
- Presencia de meeple (True/False, 2 clases)
- Posición del meeple (0-8, 9 clases, o -1 si no hay meeple)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Tuple


class CarcassonneCNN(nn.Module):
    """
    Red Neuronal Convolucional Multi-tarea para clasificar losetas de Carcassonne.
    
    Usa Transfer Learning con EfficientNet-B0 como backbone y múltiples cabezales
    de clasificación para cada tarea.
    """
    
    def __init__(
        self,
        num_tile_types: int = 25,  # A-X (24) + BLANCO (1)
        num_rotations: int = 4,     # 0, 90, 180, 270 grados
        num_meeple_classes: int = 2,  # Con meeple / Sin meeple
        num_meeple_positions: int = 9,  # Posiciones 0-8
        num_meeple_colors: int = 2,  # blue, black
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Inicializa el modelo.
        
        Args:
            num_tile_types: Número de tipos de losetas
            num_rotations: Número de rotaciones posibles
            num_meeple_classes: Clases de meeple (2: con/sin)
            num_meeple_positions: Posiciones posibles del meeple
            num_meeple_colors: Colores de meeple (2: blue/black)
            backbone: Arquitectura base ('efficientnet_b0', 'resnet18', 'resnet34', 'resnet50')
            pretrained: Si usar pesos preentrenados
            dropout: Tasa de dropout
        """
        super(CarcassonneCNN, self).__init__()
        
        self.num_tile_types = num_tile_types
        self.num_rotations = num_rotations
        self.num_meeple_classes = num_meeple_classes
        self.num_meeple_positions = num_meeple_positions
        self.num_meeple_colors = num_meeple_colors
        
        # Crear backbone según el tipo especificado
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # Remover clasificador original
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone '{backbone}' no soportado")
        
        # Capa compartida para extraer características
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512)
        )
        
        # Cabezal 1: Clasificación de tipo de loseta (A-X + BLANCO)
        self.tile_type_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_tile_types)
        )
        
        # Cabezal 2: Clasificación de rotación (0-3)
        self.rotation_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_rotations)
        )
        
        # Cabezal 3: Clasificación de presencia de meeple (binario)
        self.meeple_presence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_meeple_classes)
        )
        
        # Cabezal 4: Clasificación de posición del meeple (0-8)
        # Solo se activa si hay meeple presente
        self.meeple_position_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_meeple_positions)
        )
        
        # Cabezal 5: Clasificación de color del meeple (blue/black)
        # Solo se activa si hay meeple presente
        self.meeple_color_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_meeple_colors)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass del modelo.
        
        Args:
            x: Tensor de entrada con shape (batch_size, 3, H, W)
            
        Returns:
            Diccionario con logits para cada tarea:
            - 'tile_type': (batch_size, num_tile_types)
            - 'rotation': (batch_size, num_rotations)
            - 'meeple_presence': (batch_size, num_meeple_classes)
            - 'meeple_position': (batch_size, num_meeple_positions)
            - 'meeple_color': (batch_size, num_meeple_colors)
        """
        # Extraer características con el backbone
        features = self.backbone(x)
        
        # Pasar por la capa compartida
        shared_features = self.shared_fc(features)
        
        # Obtener predicciones de cada cabezal
        tile_type_logits = self.tile_type_head(shared_features)
        rotation_logits = self.rotation_head(shared_features)
        meeple_presence_logits = self.meeple_presence_head(shared_features)
        meeple_position_logits = self.meeple_position_head(shared_features)
        meeple_color_logits = self.meeple_color_head(shared_features)
        
        return {
            'tile_type': tile_type_logits,
            'rotation': rotation_logits,
            'meeple_presence': meeple_presence_logits,
            'meeple_position': meeple_position_logits,
            'meeple_color': meeple_color_logits
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Realiza predicciones con el modelo.
        
        Args:
            x: Tensor de entrada con shape (batch_size, 3, H, W)
            
        Returns:
            Diccionario con predicciones:
            - 'tile_type': Índice de clase predicha
            - 'rotation': Rotación predicha (0-3)
            - 'meeple_presence': 0 o 1
            - 'meeple_position': Posición predicha (0-8 o -1 si no hay meeple)
            - 'meeple_color': Color predicho (0=blue, 1=black, o -1 si no hay meeple)
            - 'confidence': Diccionario con confianza para cada predicción
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            # Obtener predicciones
            tile_type_pred = torch.argmax(logits['tile_type'], dim=1)
            rotation_pred = torch.argmax(logits['rotation'], dim=1)
            meeple_presence_pred = torch.argmax(logits['meeple_presence'], dim=1)
            meeple_position_pred = torch.argmax(logits['meeple_position'], dim=1)
            meeple_color_pred = torch.argmax(logits['meeple_color'], dim=1)
            
            # Calcular confianzas (probabilidades)
            tile_type_conf = F.softmax(logits['tile_type'], dim=1).max(dim=1)[0]
            rotation_conf = F.softmax(logits['rotation'], dim=1).max(dim=1)[0]
            meeple_presence_conf = F.softmax(logits['meeple_presence'], dim=1).max(dim=1)[0]
            meeple_position_conf = F.softmax(logits['meeple_position'], dim=1).max(dim=1)[0]
            meeple_color_conf = F.softmax(logits['meeple_color'], dim=1).max(dim=1)[0]
            
            # Si no hay meeple, la posición y color son -1
            meeple_position_final = torch.where(
                meeple_presence_pred == 0,
                torch.tensor(-1, device=x.device),
                meeple_position_pred
            )
            
            meeple_color_final = torch.where(
                meeple_presence_pred == 0,
                torch.tensor(-1, device=x.device),
                meeple_color_pred
            )
            
            return {
                'tile_type': tile_type_pred,
                'rotation': rotation_pred,
                'meeple_presence': meeple_presence_pred,
                'meeple_position': meeple_position_final,
                'meeple_color': meeple_color_final,
                'confidence': {
                    'tile_type': tile_type_conf,
                    'rotation': rotation_conf,
                    'meeple_presence': meeple_presence_conf,
                    'meeple_position': meeple_position_conf,
                    'meeple_color': meeple_color_conf
                }
            }


class MultiTaskLoss(nn.Module):
    """
    Función de pérdida multi-tarea con ponderación automática.
    
    Combina las pérdidas de todas las tareas usando ponderación dinámica
    para balancear su importancia durante el entrenamiento.
    """
    
    def __init__(
        self,
        tile_type_weight: float = 2.0,
        rotation_weight: float = 1.0,
        meeple_presence_weight: float = 1.5,
        meeple_position_weight: float = 1.0,
        meeple_color_weight: float = 1.0,
        use_label_smoothing: bool = True,
        label_smoothing: float = 0.1
    ):
        """
        Inicializa la función de pérdida.
        
        Args:
            tile_type_weight: Peso para la pérdida de tipo de loseta
            rotation_weight: Peso para la pérdida de rotación
            meeple_presence_weight: Peso para la pérdida de presencia de meeple
            meeple_position_weight: Peso para la pérdida de posición del meeple
            meeple_color_weight: Peso para la pérdida de color del meeple
            use_label_smoothing: Si usar label smoothing
            label_smoothing: Factor de suavizado de etiquetas
        """
        super(MultiTaskLoss, self).__init__()
        
        self.tile_type_weight = tile_type_weight
        self.rotation_weight = rotation_weight
        self.meeple_presence_weight = meeple_presence_weight
        self.meeple_position_weight = meeple_position_weight
        self.meeple_color_weight = meeple_color_weight
        
        smoothing = label_smoothing if use_label_smoothing else 0.0
        
        # Criterios de pérdida
        self.tile_type_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.rotation_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.meeple_presence_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.meeple_position_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, ignore_index=-1)
        self.meeple_color_criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, ignore_index=-1)
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calcula la pérdida total.
        
        Args:
            predictions: Diccionario con predicciones del modelo
            targets: Diccionario con etiquetas verdaderas
            
        Returns:
            - Pérdida total
            - Diccionario con pérdidas individuales
        """
        # Calcular pérdidas individuales
        tile_type_loss = self.tile_type_criterion(predictions['tile_type'], targets['tile_type'])
        rotation_loss = self.rotation_criterion(predictions['rotation'], targets['rotation'])
        meeple_presence_loss = self.meeple_presence_criterion(
            predictions['meeple_presence'], 
            targets['meeple_presence']
        )
        
        # Para meeple_position y meeple_color, solo calcular la pérdida si hay meeple presente
        meeple_position_loss = self.meeple_position_criterion(
            predictions['meeple_position'],
            targets['meeple_position']
        )
        
        meeple_color_loss = self.meeple_color_criterion(
            predictions['meeple_color'],
            targets['meeple_color']
        )
        
        # Pérdida total ponderada
        total_loss = (
            self.tile_type_weight * tile_type_loss +
            self.rotation_weight * rotation_loss +
            self.meeple_presence_weight * meeple_presence_loss +
            self.meeple_position_weight * meeple_position_loss +
            self.meeple_color_weight * meeple_color_loss
        )
        
        # Diccionario con pérdidas individuales (para logging)
        loss_dict = {
            'total': total_loss,
            'tile_type': tile_type_loss,
            'rotation': rotation_loss,
            'meeple_presence': meeple_presence_loss,
            'meeple_position': meeple_position_loss,
            'meeple_color': meeple_color_loss
        }
        
        return total_loss, loss_dict


def create_model(config: Dict = None) -> CarcassonneCNN:
    """
    Factory function para crear el modelo.
    
    Args:
        config: Diccionario con configuración del modelo
        
    Returns:
        Modelo CarcassonneCNN
    """
    if config is None:
        config = {}
    
    return CarcassonneCNN(
        num_tile_types=config.get('num_tile_types', 25),
        num_rotations=config.get('num_rotations', 4),
        num_meeple_classes=config.get('num_meeple_classes', 2),
        num_meeple_positions=config.get('num_meeple_positions', 9),
        num_meeple_colors=config.get('num_meeple_colors', 2),
        backbone=config.get('backbone', 'efficientnet_b0'),
        pretrained=config.get('pretrained', True),
        dropout=config.get('dropout', 0.3)
    )


if __name__ == "__main__":
    # Test del modelo
    print("=== Test del Modelo CarcassonneCNN ===\n")
    
    # Crear modelo
    model = create_model()
    print(f"Modelo creado: {model.__class__.__name__}")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}\n")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {dummy_input.shape}")
    
    outputs = model(dummy_input)
    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test predict
    predictions = model.predict(dummy_input)
    print("\nPredicciones:")
    for key, value in predictions.items():
        if key != 'confidence':
            print(f"  {key}: {value}")
    print(f"  confidence: {predictions['confidence']}")
