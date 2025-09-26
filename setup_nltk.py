import nltk
import os

def setup_nltk_data():
    """Setup NLTK data for Lambda environment"""
    nltk_data_dir = '/tmp/nltk_data'
    os.environ['NLTK_DATA'] = nltk_data_dir
    nltk.data.path.clear()
    nltk.data.path.append(nltk_data_dir)
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    resources = ['stopwords', 'wordnet', 'punkt', 'omw-1.4']
    
    for resource in resources:
        try:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            print(f'✅ Downloaded {resource}')
        except Exception as e:
            print(f'❌ Failed to download {resource}: {e}')
    
    print('NLTK setup completed')

if __name__ == '__main__':
    setup_nltk_data()