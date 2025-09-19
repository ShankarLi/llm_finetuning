from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import glob
import json
import logging
import os
import pickle
from datetime import datetime
import numpy as np

import joblib
from swagger_config import swagger_config

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ProductionSentimentAnalyzer:
    """Production-ready sentiment analyzer with comprehensive monitoring"""
    
    def __init__(self, model_path, metadata_path):
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load label encoder
        with open('output/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.requests_count = 0
        self.prediction_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Model initialized - Version: {self.metadata['timestamp']}")
    
    def predict(self, text):
        """Production-grade prediction with comprehensive error handling"""
        try:
            if not text or not text.strip():
                return {'error': 'Empty text provided'}
            
            # The model pipeline includes preprocessing
            prediction = self.model.predict([text])[0]
            probabilities = self.model.predict_proba([text])[0]
            confidence = float(max(probabilities))
            
            # Convert encoded prediction back to original label
            sentiment = self.label_encoder.inverse_transform([prediction])[0]
            
            # Ensure sentiment is a string (not numpy string)
            sentiment = str(sentiment)
            
            # Log prediction for monitoring
            self.requests_count += 1
            prediction_log = {
                'timestamp': datetime.now().isoformat(),
                'text_length': len(text),
                'prediction': sentiment,
                'confidence': confidence
            }
            
            self.prediction_history.append(prediction_log)
            if len(self.prediction_history) > 1000:
                self.prediction_history.pop(0)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'model_version': self.metadata['timestamp'],
                'probabilities': {
                    str(class_name): float(prob) 
                    for class_name, prob in zip(self.label_encoder.classes_, probabilities)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def batch_predict(self, texts):
        """Batch prediction for multiple texts"""
        try:
            if not texts or not isinstance(texts, list):
                return {'error': 'Invalid input: expected list of texts'}
            
            results = []
            for text in texts:
                result = self.predict(text)
                results.append(result)
            
            return {'predictions': results, 'count': len(results)}
            
        except Exception as e:
            self.logger.error(f"Batch prediction error: {str(e)}")
            return {'error': f'Batch prediction failed: {str(e)}'}
    
    def get_model_info(self):
        """Get comprehensive model information"""
        return {
            'model_metadata': self.metadata,
            'requests_served': self.requests_count,
            'available_classes': [str(cls) for cls in self.label_encoder.classes_],
            'last_prediction_time': self.prediction_history[-1]['timestamp'] if self.prediction_history else None,
            'model_performance': {
                'test_accuracy': self.metadata.get('test_metrics', {}).get('accuracy', 'N/A'),
                'test_f1_macro': self.metadata.get('test_metrics', {}).get('f1_macro', 'N/A'),
                'validation_f1': self.metadata.get('validation_metrics', {}).get('f1_score', 'N/A')
            }
        }
    
    def get_prediction_stats(self):
        """Get prediction statistics for monitoring"""
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}
        
        sentiments = [pred['prediction'] for pred in self.prediction_history]
        confidences = [pred['confidence'] for pred in self.prediction_history]
        
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        
        return {
            'total_predictions': len(self.prediction_history),
            'sentiment_distribution': dict(sentiment_counts),
            'average_confidence': sum(confidences) / len(confidences),
            'recent_predictions': self.prediction_history[-10:]  # Last 10 predictions
        }

# Flask App
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5002", "http://127.0.0.1:5002", "http://192.168.0.106:5002"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": True
    }
})

# Configure JSON encoder
app.json.encoder = NumpyEncoder

def get_latest_production_model():
    """Get the latest production model and metadata"""
    model_files = glob.glob('output/production_sentiment_model_*.pkl')
    if not model_files:
        return None, None
    
    latest_model = max(model_files, key=os.path.getctime)
    # Extract timestamp correctly: production_sentiment_model_20250919_182637.pkl -> 20250919_182637
    filename = os.path.basename(latest_model)
    timestamp = filename.replace('production_sentiment_model_', '').replace('.pkl', '')
    metadata_file = f'output/model_metadata_{timestamp}.json'
    
    if os.path.exists(metadata_file):
        return latest_model, metadata_file
    return latest_model, None

# Initialize analyzer
model_path, metadata_path = get_latest_production_model()
if model_path and metadata_path:
    try:
        analyzer = ProductionSentimentAnalyzer(model_path, metadata_path)
        print(f"✅ Production model loaded: {model_path}")
        print(f"📊 Model accuracy: {analyzer.metadata.get('test_metrics', {}).get('accuracy', 'N/A'):.4f}")
        print(f"🎯 Model F1-score: {analyzer.metadata.get('test_metrics', {}).get('f1_macro', 'N/A'):.4f}")
    except Exception as e:
        print(f"❌ Error loading production model: {e}")
        analyzer = None
else:
    print("⚠️ No production model found. Please train a model first.")
    analyzer = None

# Add OPTIONS handler for preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
        return response

@app.route('/')
def swagger_ui():
    """Serve Swagger UI documentation"""
    return render_template('swagger.html')

@app.route('/api-info')
def api_info():
    """API Information and Documentation"""
    return jsonify({
        "name": "Production Sentiment Analysis API",
        "version": "2.0.0",
        "description": "Advanced sentiment analysis API with machine learning optimization",
        "documentation": "Visit / for interactive API documentation",
        "endpoints": {
            "predict": "/predict (POST) - Single text sentiment prediction",
            "batch_predict": "/batch-predict (POST) - Batch sentiment prediction", 
            "model_info": "/model-info (GET) - Model information and metrics",
            "stats": "/stats (GET) - Prediction statistics",
            "health": "/health (GET) - Health check"
        },
        "example_request": {
            "url": "/predict",
            "method": "POST",
            "body": {"text": "This product is amazing!"}
        }
    })

@app.route('/swagger-spec')
def swagger_spec():
    """Serve OpenAPI specification"""
    return jsonify(swagger_config)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict_sentiment():
    """Single text sentiment prediction"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response
    
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text']
        if not analyzer:
            return jsonify({"error": "Model not loaded"}), 500

        result = analyzer.predict(text)
        if 'error' in result:
            return jsonify(result), 500
            
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch-predict', methods=['POST', 'OPTIONS'])
def batch_predict_sentiment():
    """Batch sentiment prediction"""
    if request.method == 'OPTIONS':
        response = jsonify()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response
    
    try:
        data = request.json
        if not data or 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400

        texts = data['texts']
        if not analyzer:
            return jsonify({"error": "Model not loaded"}), 500

        result = analyzer.batch_predict(texts)
        if 'error' in result:
            return jsonify(result), 500
            
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information and performance metrics"""
    if not analyzer:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify(analyzer.get_model_info())

@app.route('/stats', methods=['GET'])
def prediction_stats():
    """Get prediction statistics for monitoring"""
    if not analyzer:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify(analyzer.get_prediction_stats())

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    return jsonify({
        "status": "healthy" if analyzer else "model_not_loaded",
        "model_loaded": analyzer is not None,
        "model_path": model_path if model_path else "No model found",
        "requests_served": analyzer.requests_count if analyzer else 0,
        "model_version": analyzer.metadata['timestamp'] if analyzer else None,
        "uptime": "Available"
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 STARTING PRODUCTION SENTIMENT ANALYSIS API")
    print("="*60)
    print("📖 API Documentation: http://localhost:5002/")
    print("🔍 Health Check: http://localhost:5002/health")
    print("📊 Model Info: http://localhost:5002/model-info")
    print("📈 Statistics: http://localhost:5002/stats")
    print("💡 CORS enabled for Swagger UI")
    print("="*60)
    
    try:
        app.run(debug=False, host="0.0.0.0", port=5002)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()