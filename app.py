from flask import Flask, render_template, request, jsonify, send_file
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from main import process_board_image
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Crear carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join("modules", "imagen_generator", "output"), exist_ok=True)

# Almacenar sesiones de procesamiento
processing_sessions = {}

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
        
        # Guardar sesión para selección de losetas de referencia
        session_id = timestamp
        processing_sessions[session_id] = {
            'filepath': filepath,
            'timestamp': timestamp,
            'temp_filename': temp_filename,
            'stage': 'select_reference_tiles'
        }
        
        # Retornar para que el usuario seleccione las 8 losetas de referencia
        return jsonify({
            'needs_reference_selection': True,
            'session_id': session_id,
            'image_url': f'/uploads/{temp_filename}'
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error al procesar: {str(e)}'}), 500

@app.route('/process_with_references', methods=['POST'])
def process_with_references():
    """Procesa el tablero con las coordenadas de referencia seleccionadas por el usuario"""
    try:
        data = request.json
        session_id = data.get('session_id')
        reference_coords = data.get('reference_coords', [])  # Lista de {x, y, width, height}
        
        if session_id not in processing_sessions:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        if len(reference_coords) < 8:
            return jsonify({'error': 'Se requieren 8 losetas de referencia'}), 400
        
        session = processing_sessions[session_id]
        filepath = session['filepath']
        timestamp = session['timestamp']
        
        # Guardar coordenadas en sesión para usar después
        session['reference_coords'] = reference_coords
        
        # Procesar el tablero usando la función de main.py
        result = process_board_image(
            filepath,
            timestamp=timestamp,
            tiles_texture_pack="tiles_texture_pack-v3",
            web_mode=True,
            reference_coords=reference_coords
        )
        
        # result puede ser:
        # 1. Una tupla (dict_confirmacion, None, None) si necesita confirmaciones
        # 2. Una tupla (scores_dict, image_path, board_game) si procesó correctamente
        
        # Desempaquetar la tupla
        first_element, image_path, board_game = result
        
        # Verificar si el primer elemento es un dict de confirmación
        if isinstance(first_element, dict) and first_element.get('needs_confirmation', False):
            # Actualizar sesión
            processing_sessions[session_id]['stage'] = 'confirm_tiles'
            
            # Preparar datos de losetas que necesitan confirmación
            pending_tiles = []
            for tile_info in first_element['pending_tiles']:
                # Leer imagen de la loseta y codificarla en base64
                with open(tile_info['tile_image_path'], 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Cargar imágenes de referencia para las opciones
                ref_folder = os.path.join("modules", "piplineFotoRecorteMeepleTipo", "referencias_organizadas")
                options_with_images = []
                
                for option in tile_info['options']:
                    letter = option['letter']
                    ref_img_path = os.path.join(ref_folder, letter, f"{letter}_ref_001.png")
                    
                    if os.path.exists(ref_img_path):
                        with open(ref_img_path, 'rb') as ref_file:
                            ref_data = base64.b64encode(ref_file.read()).decode('utf-8')
                    else:
                        ref_data = None
                    
                    options_with_images.append({
                        'letter': letter,
                        'confidence': option['confidence'],
                        'reference_image': ref_data
                    })
                
                pending_tiles.append({
                    'tile_index': tile_info['tile_index'],
                    'tile_image': img_data,
                    'options': options_with_images,
                    'grid_position': tile_info['grid_position']
                })
            
            return jsonify({
                'needs_confirmation': True,
                'session_id': session_id,
                'pending_tiles': pending_tiles
            })
        
        # Si no necesita confirmación, first_element es el diccionario de scores
        scores = first_element
        
        if scores is None:
            return jsonify({'error': 'No se pudo procesar el tablero. Verifique la imagen.'}), 400
        
        player1 = scores[1]
        player2 = scores[2]
        
        # Limpiar sesión
        del processing_sessions[session_id]
        
        # Preparar respuesta
        result = {
            'success': True,
            'scores': {
                'player1': {
                    'total': player1,
                    'fields': scores['player1_fields'],
                    'features': scores['player1_features']
                },
                'player2': {
                    'total': player2,
                    'fields': scores['player2_fields'],
                    'features': scores['player2_features']
                }
            },
            'processed_image': f'/result/{timestamp}',
            'original_image': f'/uploads/{session["temp_filename"]}'
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error al procesar: {str(e)}'}), 500

@app.route('/confirm_tiles', methods=['POST'])
def confirm_tiles():
    """Endpoint para recibir confirmaciones de losetas del usuario"""
    try:
        data = request.json
        session_id = data.get('session_id')
        selections = data.get('selections', {})  # {tile_index: selected_letter}
        
        if session_id not in processing_sessions:
            return jsonify({'error': 'Sesión no encontrada'}), 404
        
        session = processing_sessions[session_id]
        filepath = session['filepath']
        timestamp = session['timestamp']
        
        # Obtener las coordenadas de referencia de la sesión (si existen)
        reference_coords = session.get('reference_coords', None)
        
        # Convertir keys a int
        manual_selections = {int(k): v for k, v in selections.items()}
        
        # Re-procesar con las selecciones manuales usando la función de main.py
        result = process_board_image(
            filepath,
            timestamp=timestamp,
            tiles_texture_pack="tiles_texture_pack-v3",
            web_mode=True,
            reference_coords=reference_coords,
            manual_selections=manual_selections
        )
        
        # Desempaquetar la tupla
        first_element, image_path, board_game = result
        
        # Verificar si aún hay confirmaciones pendientes
        if isinstance(first_element, dict) and first_element.get('needs_confirmation', False):
            # Actualizar sesión
            processing_sessions[session_id]['stage'] = 'confirm_tiles'
            
            # Retornar nuevas confirmaciones pendientes (mismo formato que antes)
            pending_tiles = []
            for tile_info in first_element['pending_tiles']:
                with open(tile_info['tile_image_path'], 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                ref_folder = os.path.join("modules", "piplineFotoRecorteMeepleTipo", "referencias_organizadas")
                options_with_images = []
                
                for option in tile_info['options']:
                    letter = option['letter']
                    ref_img_path = os.path.join(ref_folder, letter, f"{letter}_ref_001.png")
                    
                    if os.path.exists(ref_img_path):
                        with open(ref_img_path, 'rb') as ref_file:
                            ref_data = base64.b64encode(ref_file.read()).decode('utf-8')
                    else:
                        ref_data = None
                    
                    options_with_images.append({
                        'letter': letter,
                        'confidence': option['confidence'],
                        'reference_image': ref_data
                    })
                
                pending_tiles.append({
                    'tile_index': tile_info['tile_index'],
                    'tile_image': img_data,
                    'options': options_with_images,
                    'grid_position': tile_info['grid_position']
                })
            
            return jsonify({
                'needs_confirmation': True,
                'pending_tiles': pending_tiles
            })
        
        # Todas las confirmaciones completadas, first_element es el diccionario de scores
        scores = first_element
        
        # Limpiar archivos temporales de losetas
        temp_tiles_dir = "temp_tiles"
        if os.path.exists(temp_tiles_dir):
            for temp_file in os.listdir(temp_tiles_dir):
                if temp_file.startswith('temp_tile_') and temp_file.endswith('.png'):
                    try:
                        os.remove(os.path.join(temp_tiles_dir, temp_file))
                    except:
                        pass
        
        player1 = scores[1]
        player2 = scores[2]
        
        # Obtener nombre de archivo original
        temp_filename = session.get('temp_filename', f'tablero_{timestamp}.jpg')
        
        # Limpiar sesión
        del processing_sessions[session_id]
        
        # Preparar respuesta final
        return jsonify({
            'success': True,
            'scores': {
                'player1': {
                    'total': player1,
                    'fields': scores['player1_fields'],
                    'features': scores['player1_features']
                },
                'player2': {
                    'total': player2,
                    'fields': scores['player2_fields'],
                    'features': scores['player2_features']
                }
            },
            'processed_image': f'/result/{timestamp}',
            'original_image': f'/uploads/{temp_filename}'
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error al confirmar: {str(e)}'}), 500

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

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_file(os.path.join('static', filename))
    except Exception as e:
        return jsonify({'error': 'Archivo no encontrado'}), 404

if __name__ == '__main__':
    # Desactivar use_reloader en Windows para evitar problemas con el debugger
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)