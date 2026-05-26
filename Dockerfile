# ─────────────────────────────────────────────────────────
# MRI Brain Tumor Classifier — Backend API
# Multi-stage build for smaller final image
# ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (for Pillow)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg62-turbo-dev \
        zlib1g-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy model files
COPY model/ model/

# Expose port
EXPOSE 8000

# Start server — PORT is read via os.environ in app.py
CMD ["python", "app.py"]
