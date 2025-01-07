# Base image with NVIDIA CUDA 12.4 support
FROM nvidia/cuda:12.4.0-base-ubuntu20.04

# Set environment variables to prevent Python from buffering output
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip and upgrade it
RUN pip install --no-cache-dir --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the application code to the container
COPY app.py /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi uvicorn transformers torch torchvision torchaudio librosa \
    --index-url https://download.pytorch.org/whl/cu124

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]