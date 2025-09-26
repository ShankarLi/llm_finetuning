# Production stage
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies first
RUN yum update -y && yum install -y gcc

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt --target ${LAMBDA_TASK_ROOT}

# Copy NLTK setup script and run it
COPY setup_nltk.py ${LAMBDA_TASK_ROOT}/
RUN cd ${LAMBDA_TASK_ROOT} && python3 setup_nltk.py

# Copy application files (but exclude unnecessary files)
COPY *.py ${LAMBDA_TASK_ROOT}/
COPY output/ ${LAMBDA_TASK_ROOT}/output/
COPY templates/ ${LAMBDA_TASK_ROOT}/templates/

# Set environment variables
ENV NLTK_DATA=/tmp/nltk_data
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

# Set handler
CMD ["lambda_adapter.lambda_handler"]