import logging
import joblib
import datetime
import re
import numpy as np
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def save_model(model):
    """Save the trained model to output directory"""
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"output/sentiment_model_{timestamp}.pkl"
    
    try:
        joblib.dump(model, model_filename)
        logging.info(f"Model saved to {model_filename}")
        return model_filename
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        raise

def load_model(model_filename):
    """Load a trained model"""
    try:
        model = joblib.load(model_filename)
        logging.info(f"Model loaded from {model_filename}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

class SentimentAnalyzer:
    """Sentiment analysis prediction class"""
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.requests_count = 0
        self.last_predictions = []
        
    def preprocess_input(self, text):
        """Preprocess input text"""
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    def predict(self, text):
        """Predict sentiment for given text"""
        try:
            if isinstance(text, str):
                processed_text = self.preprocess_input(text)
                result = self.model.predict([processed_text])[0]
                prob = np.max(self.model.predict_proba([processed_text]))
            else:
                processed_texts = [self.preprocess_input(t) for t in text]
                result = self.model.predict(processed_texts)
                prob = np.max(self.model.predict_proba(processed_texts), axis=1)
            
            # Log request for monitoring
            self.requests_count += 1
            if len(self.last_predictions) >= 100:
                self.last_predictions.pop(0)
            self.last_predictions.append(result)
            
            return {'sentiment': result, 'confidence': float(prob)}
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {'error': str(e)}