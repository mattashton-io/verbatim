FROM python:3.13-slim

# Install system dependencies (ffmpeg is required for pydub/audio conversion)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt 
CMD ["python3", "app.py"]