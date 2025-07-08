FROM python:3.10-slim

WORKDIR /app

# 🧱 Dépendances système nécessaires à OpenCV (dont libGL)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY yolov8n-face.pt ./

RUN pip install --no-cache-dir \
    flask \
    opencv-python \
    ultralytics \
    python-dotenv

CMD ["python", "main.py"]
