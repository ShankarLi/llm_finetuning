from flask import Flask, request, jsonify, render_template
import os
import glob
from llm_finetuning.non_prod_training.deploy_model import SentimentAnalyzer
from swagger_config import swagger_config

app = Flask(__name__)

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)


# Find the latest model in output directory
def get_latest_model():
    model_files = glob.glob("output/sentiment_model_*.pkl")
    if model_files:
        return max(model_files, key=os.path.getctime)
    return None


# Load the model
model_path = get_latest_model()
if model_path:
    try:
        analyzer = SentimentAnalyzer(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        analyzer = None
else:
    print("No model found in output directory")
    analyzer = None


# Main route - serves Swagger UI
@app.route("/")
def swagger_ui():
    return render_template("swagger.html")


# Route to serve the OpenAPI specification
@app.route("/swagger-spec")
def swagger_spec():
    return jsonify(swagger_config)


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict_sentiment():
    try:
        data = request.json
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]
        if not analyzer:
            return jsonify({"error": "Model not loaded"}), 500

        result = analyzer.predict(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": analyzer is not None,
            "model_path": model_path if model_path else "No model found",
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
