FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY data_manager.py .
COPY hyperparameter_tuner.py .
COPY train_production_model.py .
COPY production_api.py .
COPY swagger_config.py .
COPY templates/ ./templates/
COPY output/ ./output/

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"

# Create directories
RUN mkdir -p logs data_cache

# Expose the port
EXPOSE 5000

# Use Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "production_api:app"]