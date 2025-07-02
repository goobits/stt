FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY install/requirements-server.txt /app/install/
RUN pip install --no-cache-dir -r install/requirements-server.txt

# Copy the entire project
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/ssl

# Generate SSL certificates if needed
RUN python install/utils/generate_ssl_certs.py

# Expose WebSocket port
EXPOSE 8769

# Set environment variables
ENV PYTHONPATH=/app
ENV STT_DOCKER_MODE=1

# Run the server
CMD ["python", "server.py", "start", "--verbose"]