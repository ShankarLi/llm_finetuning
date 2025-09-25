# Production stage
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies first
RUN yum update -y && yum install -y gcc

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# Download NLTK data to the correct location
RUN python3 -c "
import nltk
import os
# Set NLTK data path
os.environ['NLTK_DATA'] = '/tmp/nltk_data'
nltk.data.path.clear()
nltk.data.path.append('/tmp/nltk_data')
# Create directory
os.makedirs('/tmp/nltk_data', exist_ok=True)
# Download data
try:
    nltk.download('stopwords', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('wordnet', download_dir='/tmp/nltk_data', quiet=True) 
    nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('omw-1.4', download_dir='/tmp/nltk_data', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download error: {e}')
"

# Copy application files (but exclude unnecessary files)
COPY *.py ${LAMBDA_TASK_ROOT}/
COPY output/ ${LAMBDA_TASK_ROOT}/output/
COPY templates/ ${LAMBDA_TASK_ROOT}/templates/

# Set environment variables
ENV NLTK_DATA=/tmp/nltk_data
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

# Set handler
CMD ["lambda_adapter.lambda_handler"]