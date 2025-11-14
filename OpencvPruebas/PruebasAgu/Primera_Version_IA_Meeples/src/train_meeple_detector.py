import torch
from torch.utils.data import DataLoader
from meeple_detector import MeepleDataset, MeepleCNN, MeepleTrainer, create_data_transforms
import json
import os

def main():
    # Configuración
    batch_size = 32
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Usando dispositivo: {device}")

    # Crear transformaciones
    train_transform = create_data_transforms(augment=True)
    val_transform = create_data_transforms(augment=False)

    # Cargar datasets
    train_dataset = MeepleDataset('data/train_annotations.json', transform=train_transform)
    val_dataset = MeepleDataset('data/val_annotations.json', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train dataset: {len(train_dataset)} muestras")
    print(f"Val dataset: {len(val_dataset)} muestras")

    # Crear modelo
    model = MeepleCNN()

    # Crear entrenador
    trainer = MeepleTrainer(model, device=device)

    # Entrenar
    trainer.train(train_loader, val_loader, epochs=epochs)

    # Guardar historial
    with open('output/training_history.json', 'w') as f:
        json.dump(trainer.history, f, indent=2)

    print("Entrenamiento completado. Modelo guardado en models/best_meeple_model.pth")

if __name__ == "__main__":
    # Verificar que existan los archivos de anotaciones
    if not os.path.exists('data/train_annotations.json'):
        print("❌ No se encontró data/train_annotations.json")
        print("Crea anotaciones de entrenamiento primero.")
        exit(1)

    if not os.path.exists('data/val_annotations.json'):
        print("❌ No se encontró data/val_annotations.json")
        print("Crea anotaciones de validación primero.")
        exit(1)

    main()