# Use an official Python runtime as a parent image
# Python 3.10 is recommended for compatibility with AI libraries
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for audio processing in a single layer
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Hugging Face cache directory to a temporary, writable location
ENV HF_HOME=/tmp/hf_cache
ENV HF_HUB_DISABLE_SYMLINKS_HF_HOME=1

# Create the cache directory with proper permissions
RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE $PORT

# Health check (optional but recommended for production)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn merged_backend:app --host 0.0.0.0 --port ${PORT:-7860}"]
