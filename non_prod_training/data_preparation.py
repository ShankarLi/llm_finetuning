import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

def download_nltk_resources():
    try:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        logging.info("NLTK resources downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {str(e)}")

# Text preprocessing
def preprocess_text(df, text_column='review'):
    logging.info("Starting text preprocessing")
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    
    df['cleaned_review'] = df[text_column].apply(clean_text)
    logging.info("Text preprocessing completed")
    return df