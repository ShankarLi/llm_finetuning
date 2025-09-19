# Production Sentiment Analysis System

A comprehensive, production-ready machine learning pipeline for sentiment analysis featuring advanced hyperparameter optimization, experiment tracking, and RESTful API deployment.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION SENTIMENT ANALYSIS SYSTEM         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Data Layer    │  │  Training Layer │  │  Serving Layer  │ │
│  │                 │  │                 │  │                 │ │
│  │ • HuggingFace   │  │ • Optuna        │  │ • Flask API     │ │
│  │ • Preprocessing │  │ • MLflow        │  │ • Swagger UI    │ │
│  │ • Caching       │  │ • Validation    │  │ • Monitoring    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│            │                    │                    │         │
│            └────────────────────┼────────────────────┘         │
│                                 │                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Storage & Monitoring                     │   │
│  │ • Model Artifacts  • Experiment Logs  • Metrics       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Machine Learning Pipeline

### Data Flow Diagram
```
HuggingFace Dataset → Text Preprocessing → Feature Extraction → Model Training → Evaluation → Deployment
       ↓                      ↓                    ↓              ↓           ↓          ↓
   • Cache Data         • Clean Text        • TF-IDF         • Optuna      • Test     • REST API
   • Validation         • Tokenization      • N-grams        • 5-Fold CV   • Metrics  • Monitoring
   • Train/Val/Test     • Lemmatization     • Vectorization  • MLflow      • Reports  • Swagger
```

### Algorithm Components

#### 1. **Text Preprocessing Pipeline**
- **URL/Email Removal**: Regular expressions to clean social media artifacts
- **Tokenization**: NLTK word tokenization
- **Lemmatization**: WordNet lemmatizer for word normalization
- **Stop Word Removal**: English stop words filtering
- **N-gram Generation**: Unigram, bigram, trigram feature extraction

#### 2. **Feature Engineering**
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **Feature Selection**: Max features optimization (1K-10K range)
- **Document Frequency Filtering**: Min/max document frequency thresholds
- **Sparse Matrix Optimization**: Memory-efficient storage

#### 3. **Machine Learning Models**
- **Primary**: Logistic Regression with L2 regularization
- **Solver Options**: LBFGS (small datasets), SAGA (large datasets)
- **Class Balancing**: Automatic class weight adjustment
- **Hyperparameter Space**:
  ```python
  {
      'max_features': (1000, 10000),
      'ngram_range': [(1,1), (1,2), (1,3)],
      'min_df': (1, 10),
      'max_df': (0.7, 1.0),
      'C': (0.01, 100),  # Log scale
      'solver': ['lbfgs', 'saga'],
      'max_iter': (500, 2000),
      'class_weight': [None, 'balanced']
  }
  ```

#### 4. **Optimization Framework**
- **Optuna TPE Sampler**: Tree-structured Parzen Estimator
- **Bayesian Optimization**: Intelligent hyperparameter search
- **Pruning**: Median pruner for early stopping
- **Cross-Validation**: 5-fold stratified CV for robust evaluation
- **Multi-objective**: F1-macro score optimization

#### 5. **Experiment Tracking**
- **MLflow Integration**: Experiment logging and model versioning
- **Artifact Storage**: Model serialization and metadata
- **Metric Tracking**: Performance metrics across trials
- **Parameter Logging**: Hyperparameter history

## 🚀 Quick Start Guide

### Prerequisites
```bash
# System Requirements
Python 3.9+
4GB+ RAM
2GB+ Disk Space
```

### Installation
```bash
# 1. Clone Repository
git clone https://github.com/your-repo/sentiment-analysis
cd sentiment-analysis

# 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Verify Installation
python -c "import nltk, sklearn, optuna, mlflow; print('✅ All dependencies installed')"
```

### Training the Model
```bash
# Start Complete Training Pipeline
python train_production_model.py

# Expected Output:
# ✅ Dataset loaded: 30,000+ samples
# 🔧 Optimization: 75 trials
# 📊 Best F1-score: ~0.697
# ⏱️ Training time: ~45 minutes
# 💾 Model saved: output/production_sentiment_model_*.pkl
```

### Starting the API
```bash
# Production Server
python production_api.py

# Expected Output:
# ✅ Production model loaded
# 📊 Model accuracy: 0.6940
# 🎯 Model F1-score: 0.6979
# 🚀 Server running on http://0.0.0.0:5000
```

### Using Docker
```bash
# Build Image
docker build -t sentiment-api .

# Run Container
docker run -p 5000:5000 sentiment-api

# Health Check
curl http://localhost:5000/health
```

## 📊 Model Performance

### Current Production Model Metrics
```
Test Accuracy:  69.40%
Test F1-Macro:  69.79%
Val F1-Score:   69.74%
CV Score:       68.52%

Class Distribution:
├── Positive: 40.0% (8,507 samples)
├── Neutral:  31.4% (6,690 samples)
└── Negative: 28.6% (6,132 samples)
```

### Optimization Results
```
Hyperparameter Search:
├── Trials Executed: 75
├── Search Space: 8 dimensions
├── Best Parameters:
│   ├── max_features: 8,049
│   ├── ngram_range: (1,1)
│   ├── min_df: 2
│   ├── max_df: 0.714
│   ├── C: 0.627
│   ├── solver: 'saga'
│   ├── max_iter: 962
│   └── class_weight: 'balanced'
└── Optimization Time: ~43 minutes
```

## 🔗 API Documentation

### Base URL
```
http://localhost:5000
```

### Authentication
Currently no authentication required (add for production deployment)

### Endpoints

#### 1. Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

# Response:
{
  "sentiment": "positive",
  "confidence": 0.945,
  "model_version": "20250919_182637",
  "probabilities": {
    "positive": 0.945,
    "negative": 0.032,
    "neutral": 0.023
  }
}
```

#### 2. Batch Prediction
```bash
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible service", "It'\''s okay"]}'
```

#### 3. Model Information
```bash
curl http://localhost:5000/model-info

# Response includes:
# • Model metadata
# • Performance metrics
# • Training parameters
# • Dataset statistics
```

#### 4. Usage Statistics
```bash
curl http://localhost:5000/stats

# Response includes:
# • Request count
# • Sentiment distribution
# • Average confidence
# • Recent predictions
```

#### 5. Health Check
```bash
curl http://localhost:5000/health
```

### Interactive Documentation
Visit `http://localhost:5000` for Swagger UI interface

## 📁 Project Structure

```
sentiment-analysis/
├── 📁 data_cache/              # Cached datasets
├── 📁 logs/                    # Application logs
├── 📁 output/                  # Model artifacts
│   ├── production_sentiment_model_*.pkl
│   ├── model_metadata_*.json
│   ├── label_encoder.pkl
│   ├── final_confusion_matrix.png
│   └── final_metrics.png
├── 📁 templates/
│   └── swagger.html            # API documentation UI
├── 📄 data_manager.py          # Data loading & preprocessing
├── 📄 hyperparameter_tuner.py  # Optuna optimization
├── 📄 train_production_model.py # Training pipeline
├── 📄 production_api.py        # Flask REST API
├── 📄 swagger_config.py        # API specification
├── 📄 requirements.txt         # Dependencies
├── 📄 Dockerfile              # Container configuration
└── 📄 README.md               # This file
```

## 🛠️ Development & Customization

### Adding New Models
```python
# In hyperparameter_tuner.py
def objective_new_model(self, trial, X_train, y_train):
    # Define hyperparameter space
    # Create model pipeline
    # Return cross-validation score
    pass
```

### Custom Preprocessing
```python
# In data_manager.py
def custom_preprocessing(self, text):
    # Add domain-specific cleaning
    # Return processed text
    pass
```

### Monitoring Integration
```python
# Add to production_api.py
@app.route('/metrics')
def prometheus_metrics():
    # Export metrics for Prometheus
    pass
```

## 🔧 Production Deployment

### Environment Variables
```bash
export FLASK_ENV=production
export MODEL_PATH=/app/models/latest
export LOG_LEVEL=INFO
export MAX_WORKERS=4
```

### Gunicorn Configuration
```bash
gunicorn --bind 0.0.0.0:5000 \
         --workers 4 \
         --timeout 120 \
         --worker-class sync \
         --max-requests 1000 \
         production_api:app
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-api
  template:
    metadata:
      labels:
        app: sentiment-api
    spec:
      containers:
      - name: sentiment-api
        image: sentiment-api:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 📈 Monitoring & Maintenance

### Performance Monitoring
- **Latency**: Response time tracking
- **Throughput**: Requests per second
- **Accuracy**: Prediction confidence monitoring
- **Resource Usage**: CPU/Memory utilization

### Model Retraining Triggers
- **Performance Degradation**: F1-score drops below 65%
- **Data Drift**: Input distribution changes
- **Scheduled**: Monthly retraining
- **New Data**: Significant dataset updates

### Logging & Alerts
```python
# Log levels and destinations
INFO    → Application events
WARNING → Performance issues  
ERROR   → Prediction failures
CRITICAL → System failures
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/ -v --coverage

# Test coverage includes:
# • Data preprocessing
# • Model predictions
# • API endpoints
# • Error handling
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 -T 'application/json' \
   -p test_data.json \
   http://localhost:5000/predict
```

### Integration Tests
```bash
# End-to-end pipeline testing
python -m pytest tests/test_integration.py
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Code formatting
black --line-length 88 .
flake8 --max-line-length 88
```

### Pull Request Process
1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

## 📞 Support & Troubleshooting

### Common Issues

#### Model Not Loading
```bash
# Check model files exist
ls -la output/production_sentiment_model_*.pkl

# Verify dependencies
pip check

# Check logs
tail -f logs/production_training.log
```

#### Memory Issues
```bash
# Reduce batch size
export BATCH_SIZE=32

# Use lighter model
export MODEL_TYPE=logistic_regression
```

#### Performance Issues
```bash
# Enable caching
export ENABLE_CACHE=true

# Increase workers
export GUNICORN_WORKERS=8
```

### Contact Information
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: admin@sentimentapi.com

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- HuggingFace for dataset hosting
- Optuna team for optimization framework
- MLflow for experiment tracking
- scikit-learn community for ML tools

---

**Built with ❤️ for production ML systems**