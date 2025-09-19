import json
import logging
import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataManager:
    """Production-ready data management class"""
    
    def __init__(self, cache_dir='data_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs('output', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        self.setup_logging()
        self.download_nltk_resources()
        
    def setup_logging(self):
        """Setup logging for data operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def download_nltk_resources(self):
        """Download required NLTK resources"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True) 
            nltk.download('punkt', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            self.logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            self.logger.error(f"Error downloading NLTK resources: {str(e)}")
    
    def load_huggingface_dataset(self, force_reload=False):
        """Load and cache HuggingFace dataset"""
        cache_file = os.path.join(self.cache_dir, 'processed_dataset.pkl')
        
        if os.path.exists(cache_file) and not force_reload:
            self.logger.info("Loading cached dataset...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        try:
            self.logger.info("Loading dataset from HuggingFace...")
            dataset = load_dataset("mteb/tweet_sentiment_extraction")
            
            # Convert to pandas
            train_df = dataset['train'].to_pandas()
            test_df = dataset['test'].to_pandas()
            
            # Analyze and standardize column names
            train_df = self._standardize_columns(train_df)
            test_df = self._standardize_columns(test_df)
            
            # Create validation set from training data
            train_df, val_df = train_test_split(
                train_df, 
                test_size=0.2, 
                random_state=42, 
                stratify=train_df['sentiment'] if 'sentiment' in train_df.columns else None
            )
            
            # Cache the processed dataset
            dataset_dict = {
                'train': train_df,
                'validation': val_df,
                'test': test_df
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(dataset_dict, f)
                
            self.logger.info(f"Dataset loaded and cached. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return dataset_dict
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _standardize_columns(self, df):
        """Standardize column names across different dataset formats"""
        # Map common column variations to standard names
        column_mapping = {
            'text': 'review',
            'tweet': 'review',
            'content': 'review',
            'label': 'sentiment',
            'target': 'sentiment',
            'class': 'sentiment'
        }
        
        df_renamed = df.rename(columns=column_mapping)
        
        # Ensure we have the required columns
        if 'review' not in df_renamed.columns:
            # Try to find text column
            text_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['text', 'tweet', 'content', 'message'])]
            if text_cols:
                df_renamed['review'] = df[text_cols[0]]
            else:
                raise ValueError("No text column found in dataset")
        
        if 'sentiment' not in df_renamed.columns:
            # Try to find label column
            label_cols = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in ['label', 'target', 'class', 'sentiment'])]
            if label_cols:
                df_renamed['sentiment'] = df[label_cols[0]]
            else:
                raise ValueError("No label column found in dataset")
        
        return df_renamed[['review', 'sentiment']]
    
    def preprocess_text(self, df, text_column='review'):
        """Advanced text preprocessing for production"""
        self.logger.info("Starting text preprocessing...")
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove mentions and hashtags (but keep the text)
            text = re.sub(r'[@#]\w+', '', text)
            
            # Remove extra whitespace and special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Tokenize
            try:
                tokens = nltk.word_tokenize(text)
                # Remove stopwords and lemmatize
                tokens = [lemmatizer.lemmatize(token) for token in tokens 
                         if token not in stop_words and len(token) > 2]
                return ' '.join(tokens)
            except Exception:
                # Fallback if tokenization fails
                words = text.split()
                return ' '.join([word for word in words if word not in stop_words and len(word) > 2])
        
        # Apply preprocessing
        df['cleaned_review'] = df[text_column].apply(clean_text)
        
        # Remove empty reviews
        initial_count = len(df)
        df = df[df['cleaned_review'].str.len() > 0].copy()
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} empty reviews after preprocessing")
        
        self.logger.info(f"Text preprocessing completed. Final dataset size: {len(df)}")
        return df
    
    def encode_labels(self, train_df, val_df, test_df):
        """Encode labels consistently across all splits"""
        self.logger.info("Encoding labels...")
        
        # Fit encoder on training data
        label_encoder = LabelEncoder()
        train_df['sentiment_encoded'] = label_encoder.fit_transform(train_df['sentiment'])
        
        # Transform validation and test sets
        val_df['sentiment_encoded'] = label_encoder.transform(val_df['sentiment'])
        test_df['sentiment_encoded'] = label_encoder.transform(test_df['sentiment'])
        
        # Save label encoder for later use
        encoder_path = 'output/label_encoder.pkl'
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        self.logger.info(f"Labels encoded. Classes: {label_encoder.classes_}")
        return train_df, val_df, test_df, label_encoder
    
    def get_data_statistics(self, train_df, val_df, test_df):
        """Generate comprehensive data statistics"""
        stats = {
            'train_size': int(len(train_df)),
            'val_size': int(len(val_df)),
            'test_size': int(len(test_df)),
            'train_sentiment_dist': {str(k): int(v) for k, v in train_df['sentiment'].value_counts().to_dict().items()},
            'val_sentiment_dist': {str(k): int(v) for k, v in val_df['sentiment'].value_counts().to_dict().items()},
            'test_sentiment_dist': {str(k): int(v) for k, v in test_df['sentiment'].value_counts().to_dict().items()},
            'avg_text_length': {
                'train': float(train_df['cleaned_review'].str.len().mean()),
                'val': float(val_df['cleaned_review'].str.len().mean()),
                'test': float(test_df['cleaned_review'].str.len().mean())
            }
        }
        
        # Save statistics
        stats_path = 'output/dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Dataset statistics saved to {stats_path}")
        return stats