# Multi-stage Dockerfile for Lexical Semantic Embedding Model
# Optimized for both CPU and GPU deployment

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app && \
    chown -R app:app /app

USER app
WORKDIR /app

# Install Python dependencies
COPY --chown=app:app requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Add user's local bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Production stage
FROM base as production

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/embeddings \
    models logs evaluation_results \
    && chmod +x scripts/*.py

# Set default command
CMD ["python", "main.py", "--help"]

# Development stage with additional tools
FROM production as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER app

# Install development Python packages
RUN pip install --user --no-cache-dir \
    jupyter \
    ipykernel \
    black \
    flake8 \
    pytest \
    pytest-cov \
    mypy

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# GPU stage based on NVIDIA CUDA
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=0

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app && \
    chown -R app:app /app

USER app
WORKDIR /app

# Copy requirements and install GPU-specific packages
COPY --chown=app:app requirements.txt .

# Install PyTorch with CUDA support
RUN pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
RUN pip install --user --no-cache-dir -r requirements.txt

# Add user's local bin to PATH
ENV PATH="/home/app/.local/bin:${PATH}"

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/embeddings \
    models logs evaluation_results \
    && chmod +x scripts/*.py

# Set default command
CMD ["python", "main.py", "--help"]

# fix: apt libgomp1 for sklearn/faiss
