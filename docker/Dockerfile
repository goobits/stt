# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-pip python3.12-dev python3.12-venv \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install system dependencies including OpenSSL for certificates and curl for health checks
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    git \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies for the Docker server with GPU support
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    faster-whisper \
    fastapi[all] \
    uvicorn \
    websockets \
    cryptography \
    pyjwt \
    httpx \
    pydantic \
    python-multipart \
    nvidia-ml-py3

# Copy the entire project from parent directory
COPY .. /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/ssl /app/data /app/data/encryption

# Generate SSL certificates
RUN openssl req -x509 -newkey rsa:4096 \
    -keyout /app/ssl/key.pem \
    -out /app/ssl/cert.pem \
    -days 365 -nodes \
    -subj "/C=US/ST=State/L=City/O=STT/OU=Server/CN=localhost" || true

# Expose both WebSocket and dashboard ports
EXPOSE 8769 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV STT_DOCKER_MODE=1
ENV WEBSOCKET_PORT=8769
ENV WEB_PORT=8080
ENV WEBSOCKET_BIND_HOST=0.0.0.0

# GPU and CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Whisper GPU settings
ENV WHISPER_DEVICE=auto
ENV WHISPER_COMPUTE_TYPE=auto

# Health check for the services
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# Run the server with dashboard enabled
CMD ["python", "server.py", "start", "--dashboard", "--verbose"]