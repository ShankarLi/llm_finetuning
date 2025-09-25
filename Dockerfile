# Production stage
FROM public.ecr.aws/lambda/python:3.9

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT} --no-cache-dir

# Download NLTK data
RUN python3 -c "import nltk; import os; os.makedirs('/tmp/nltk_data', exist_ok=True); nltk.data.path.append('/tmp/nltk_data'); nltk.download('stopwords', download_dir='/tmp/nltk_data', quiet=True); nltk.download('wordnet', download_dir='/tmp/nltk_data', quiet=True); nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True); nltk.download('omw-1.4', download_dir='/tmp/nltk_data', quiet=True)"

# Copy application files
COPY . ${LAMBDA_TASK_ROOT}

# Set handler
CMD ["lambda_adapter.lambda_handler"]