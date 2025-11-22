#!/usr/bin/env python3
"""
Script para entrenar el modelo CNN con múltiples imágenes por clase
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.piplineFotoRecorteMeepleTipo.cnn_connector import CarcassonneCNN

class CarcassonneMultiImageDataset(Dataset):
    """Dataset personalizado para múltiples imágenes por clase"""

    def __init__(self, reference_folder, transform=None):
        self.reference_folder = Path(reference_folder)
        self.transform = transform
        self.data = []

        # Mapear letras a índices (24 letras A-X + BLANCO = 25 clases)
        self.letter_to_idx = {chr(65 + i): i for i in range(24)}  # A=0, B=1, ..., X=23
        self.letter_to_idx['BLANCO'] = 24  # BLANCO=24

        # Cargar todas las imágenes de cada carpeta
        for letter in self.letter_to_idx.keys():
            letter_folder = self.reference_folder / letter
            if letter_folder.exists():
                # Buscar todas las imágenes PNG en la carpeta
                for img_file in letter_folder.glob("*.png"):
                    self.data.append((str(img_file), self.letter_to_idx[letter]))

        print(f"Dataset creado con {len(self.data)} imágenes de {len(self.letter_to_idx)} clases (A-X + BLANCO)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = cv2.imread(img_path)
        if image is None:
            # Si la imagen no se puede cargar, devolver una imagen negra
            image = np.zeros((64, 64, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model_multi():
    """Entrenar el modelo CNN con múltiples imágenes por clase"""

    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Transformaciones con data augmentation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),  # Rotación aleatoria
        transforms.RandomHorizontalFlip(),  # Flip horizontal
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variación de color
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset y DataLoader
    dataset = CarcassonneMultiImageDataset('modules/piplineFotoRecorteMeepleTipo/referencias_organizadas', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Batch size mayor

    # Modelo
    model = CarcassonneCNN(num_classes=25).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento
    num_epochs = 100  # Más epochs para más datos
    print(f"Entrenando modelo con {len(dataset)} imágenes...")
    
    # Listas para guardar métricas
    loss_history = []
    accuracy_history = []
    top2_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        top2_correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calcular accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calcular top-2 accuracy
            _, top2_pred = torch.topk(outputs.data, 2, dim=1)
            top2_correct += sum([labels[i] in top2_pred[i] for i in range(labels.size(0))])

        # Calcular métricas promedio del epoch
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        top2_accuracy = 100 * top2_correct / total
        
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        top2_accuracy_history.append(top2_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, Top-2 Acc: {top2_accuracy:.2f}%")

    # Guardar modelo
    model_path = 'modules/piplineFotoRecorteMeepleTipo/carcassonne_cnn_multi_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado como '{model_path}'")
    
    # Generar y guardar gráficos de métricas
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, num_epochs + 1)
    
    # Gráfico de pérdida
    axes[0].plot(epochs_range, loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Pérdida durante el Entrenamiento', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de accuracy
    axes[1].plot(epochs_range, accuracy_history, 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy durante el Entrenamiento', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    # Gráfico de top-2 accuracy
    axes[2].plot(epochs_range, top2_accuracy_history, 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Top-2 Accuracy (%)', fontsize=12)
    axes[2].set_title('Top-2 Accuracy durante el Entrenamiento', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Guardar gráfico
    graph_path = 'modules/piplineFotoRecorteMeepleTipo/training_metrics.png'
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico de métricas guardado como '{graph_path}'")
    print(f"Accuracy final: {accuracy_history[-1]:.2f}%")
    print(f"Top-2 Accuracy final: {top2_accuracy_history[-1]:.2f}%")
    plt.close()

if __name__ == "__main__":
    train_model_multi()