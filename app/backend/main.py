from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import ModelPredictor
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

predictor = None

def initialize_predictor():
    global predictor
    try:
        predictor = ModelPredictor()
        if not predictor.is_ready():
            raise RuntimeError("Model not ready for prediction")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        return False

@app.route('/api/predict', methods=['POST'])
def predict():
    if not predictor or not predictor.is_ready():
        return jsonify({'error': 'Model predictor not available'}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Text input is required'}), 400
    
    if len(text) > 500:
        return jsonify({'error': 'Text too long, maximum 500 characters'}), 400
    
    start_time = time.time()
    
    try:
        result = predictor.predict(text)
        processing_time = round(time.time() - start_time, 3)
        
        response = {
            'predicted_emotion': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'processing_time': processing_time,
            'model_info': result['model_info'],
            'original_text': text,
            'processed_text': result.get('processed_text', text)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/models/status', methods=['GET'])
def model_status():
    if not predictor:
        return jsonify({'error': 'Model predictor not initialized'}), 500
    
    return jsonify(predictor.get_status())

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'predictor_available': predictor is not None,
        'predictor_ready': predictor.is_ready() if predictor else False,
        'timestamp': time.time()
    })

@app.route('/api/reinitialize', methods=['POST'])
def reinitialize():
    global predictor
    predictor = None
    
    if initialize_predictor():
        return jsonify({'message': 'Predictor reinitialized successfully'})
    else:
        return jsonify({'error': 'Reinitialization failed'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Initializing FeelOut API Server...")
    
    if initialize_predictor():
        print("Server ready with baseline model")
    else:
        print("Server starting with limited functionality")
    
    print("Available endpoints:")
    print("  POST /api/predict - Analyze emotion in text")
    print("  GET /api/models/status - Check model availability")
    print("  GET /api/health - Health check")
    print("  POST /api/reinitialize - Reinitialize models")
    
    app.run(debug=True, host='0.0.0.0', port=8080)