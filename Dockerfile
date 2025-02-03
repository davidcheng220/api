FROM python:3.10-slim-bookworm

# Make working directories
RUN  mkdir -p  /trash
WORKDIR  /trash
RUN mkdir ./images

# Upgrade pip with no cache
RUN pip install --no-cache-dir -U pip

# Copy application requirements file to the created working directory
COPY requirements-latest.txt .

# Install application dependencies from the requirements file
RUN pip install --no-cache-dir --no-deps -r requirements-latest.txt

# Copy every file in the source folder to the created working directory
COPY  . .

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

EXPOSE 8080

# Run the python application
CMD ["python3", "-m", "uvicorn", "--host", "0.0.0.0", "--port", "8080", "main:app", "--reload"]