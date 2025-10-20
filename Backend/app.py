from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # habilita requests desde el front

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    # Acá iría tu modelo de IA
    result1 = "Predicción 1"
    result2 = "Predicción 2"
    return jsonify({"result1": result1, "result2": result2})

if __name__ == '__main__':
    app.run(port=5000)
