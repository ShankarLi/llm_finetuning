# Production stage
FROM public.ecr.aws/lambda/python:3.9

# Copy requirements and install
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# Download NLTK data to /tmp (Lambda writable directory)
RUN python -c " \
import nltk \
import os \
nltk.data.path.append('/tmp/nltk_data') \
os.makedirs('/tmp/nltk_data', exist_ok=True) \
nltk.download('stopwords', download_dir='/tmp/nltk_data', quiet=True) \
nltk.download('wordnet', download_dir='/tmp/nltk_data', quiet=True) \
nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True) \
nltk.download('omw-1.4', download_dir='/tmp/nltk_data', quiet=True) \
"

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda_adapter.lambda_handler"]