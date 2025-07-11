FROM python:3.10

RUN apt-get update && apt-get install -y \
    build-essential cmake libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev libboost-python-dev \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
    libmariadb-dev \
    && apt-get clean

WORKDIR /app

COPY recognizer.py ./
COPY faces_known/ ./faces_known/

RUN pip install --no-cache-dir \
    face_recognition \
    opencv-python \
    mysql-connector-python \
    python-dotenv

RUN mkdir -p /app/images/faces_unknown

#CMD ["python", "-u", "recognizer.py"]
CMD ["tail", "-f", "/dev/null"]

