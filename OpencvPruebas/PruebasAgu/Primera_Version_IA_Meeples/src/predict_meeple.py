from meeple_detector import MeeplePredictor
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Detectar meeples azules/negros en losetas de Carcassonne')
    parser.add_argument('image_path', help='Ruta a la imagen de la loseta o directorio con imágenes')
    parser.add_argument('--model', default='models/best_meeple_model.pth', help='Ruta al modelo entrenado')

    args = parser.parse_args()

    # Verificar que el modelo existe
    if not os.path.exists(args.model):
        print(f"❌ No se encontró el modelo: {args.model}")
        print("Entrena el modelo primero con: python src/train_meeple_detector.py")
        exit(1)

    # Crear predictor
    predictor = MeeplePredictor(args.model)

    # Verificar si es un directorio o archivo
    if os.path.isdir(args.image_path):
        print(f"Procesando directorio: {args.image_path}")
        results = predictor.predict_batch(args.image_path)

        # Guardar resultados
        import json
        with open('output/prediction_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Resultados guardados en output/prediction_results.json")

    elif os.path.isfile(args.image_path):
        print(f"Procesando imagen: {args.image_path}")
        result = predictor.predict(args.image_path)

        print("\nResultado:")
        print(f"Tiene meeple azul/negro: {'Sí' if result['has_blue_or_black_meeple'] else 'No'}")
        print(f"Confianza: {result['meeple_confidence']:.2%}")
        if result['has_blue_or_black_meeple']:
            print(f"Posición: {result['meeple_position']}")

        # Visualizar
        predictor.visualize_prediction(args.image_path)

    else:
        print(f"❌ Ruta no encontrada: {args.image_path}")

if __name__ == "__main__":
    main()