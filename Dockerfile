FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies (PyTorch CPU first, then others)
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY api.py .
COPY test_webcam.py .

# Create necessary directories
RUN mkdir -p data models results

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "src/train.py"]