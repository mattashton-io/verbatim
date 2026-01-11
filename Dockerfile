FROM python:3.13-slim

# Install system dependencies (ffmpeg is required for audio conversion)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY app.py .
COPY transcription_service_v1.py .
COPY secret_manager_utils.py .
COPY templates/ ./templates/

# Expose port (Cloud Run defaults to 8080, but good to document)
EXPOSE 8080

CMD ["python3", "app.py"]