from flask import Flask, render_template, request, jsonify, send_file
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from origin_matrix import Board
from modules.piplineFotoRecorteMeepleTipo.origin_matrix_converter import OriginMatrixConverter
from modules.imagen_generator.board_image_generator import BoardImageGenerator
from modules.incomplete_features_scorer.incomplete_features_scorer import GameScorer
from modules.CarcassoneFieldsv5.puntos_campos import calculate_field_scores

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Crear carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join("modules", "imagen_generator", "output"), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_board():
    try:
        # Verificar si se subió un archivo
        if 'board_image' not in request.files:
            return jsonify({'error': 'No se encontró ninguna imagen'}), 400
        
        file = request.files['board_image']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de archivo no permitido. Use JPG, JPEG o PNG'}), 400
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"tablero_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(filepath)
        
        # Procesar el tablero
        origin_convert = OriginMatrixConverter()
        board_game = origin_convert.convert(filepath)
        
        if board_game is None:
            return jsonify({'error': 'No se pudo procesar el tablero. Verifique la imagen.'}), 400
        
        # Generar imagen del tablero procesado
        tiles_img_path = os.path.join("resources", "tiles_texture_pack-v3")
        gen_images = BoardImageGenerator(tiles_img_path)
        image_path = os.path.join("modules", "imagen_generator", "output", f"tablero_{timestamp}.jpg")
        gen_images.generate_board_image(board_game, image_path)
        
        # Calcular puntos
        scores_incomplete_features_scorer = GameScorer(board_game).score()
        scores_fields = calculate_field_scores(image_path)
        
        player1 = scores_fields.get(1, 0) + scores_incomplete_features_scorer.get(1, 0)
        player2 = scores_fields.get(2, 0) + scores_incomplete_features_scorer.get(2, 0)
        
        # Preparar respuesta
        result = {
            'success': True,
            'scores': {
                'player1': {
                    'total': player1,
                    'fields': scores_fields.get(1, 0),
                    'features': scores_incomplete_features_scorer.get(1, 0)
                },
                'player2': {
                    'total': player2,
                    'fields': scores_fields.get(2, 0),
                    'features': scores_incomplete_features_scorer.get(2, 0)
                }
            },
            'processed_image': f'/result/{timestamp}',
            'original_image': f'/uploads/{temp_filename}'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'Error al procesar: {str(e)}'}), 500

@app.route('/result/<timestamp>')
def get_result_image(timestamp):
    try:
        image_path = os.path.join("modules", "imagen_generator", "output", f"tablero_{timestamp}.jpg")
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': 'Imagen no encontrada'}), 404

@app.route('/uploads/<filename>')
def get_upload_image(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': 'Imagen no encontrada'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)